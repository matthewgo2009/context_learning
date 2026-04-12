"""GRPO (Group Relative Policy Optimization) trainer for Solver model.

Algorithm:
    For each prompt x in the batch:
        1. Sample G outputs {o_1, ..., o_G} from current policy pi_theta
        2. Compute reward r_i for each output
        3. Group-relative advantage: A_i = (r_i - mean(r)) / (std(r) + eps)
        4. For each (x, o_i):
            ratio = pi_theta(o_i|x) / pi_old(o_i|x)
            clipped_ratio = clip(ratio, 1-eps, 1+eps)
            loss_i = -min(ratio * A_i, clipped_ratio * A_i)
        5. KL penalty against reference model:
            KL_i = mean(pi_ref_logprob - pi_theta_logprob)
        6. Total loss = mean(loss) + beta * mean(KL)

Reference: DeepSeek-R1 (https://arxiv.org/abs/2501.12948)
"""

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..config.default_config import merge_config
from ..data.loader import CLBenchDataLoader
from ..models.challenge_model import ChallengeModel
from ..models.solver_model import SolverModel
from ..rewards.base_reward import SolverRewardResult
from ..rewards.rubrics_reward import RubricsReward

logger = logging.getLogger(__name__)


@dataclass
class GRPOGroupResult:
    """Result of one GRPO group for a single prompt."""

    prompt_messages: List[dict] = field(default_factory=list)
    responses: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)
    input_ids: Optional[torch.Tensor] = None
    generated_ids_list: List[torch.Tensor] = field(default_factory=list)
    reward_breakdowns: List[SolverRewardResult] = field(default_factory=list)


class GRPOTrainer:
    """
    GRPO trainer that actually updates model parameters.

    For each prompt, samples G responses, computes group-relative advantages,
    and performs clipped policy gradient updates with KL regularization.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        challenge_model: Optional[ChallengeModel] = None,
        solver_model: Optional[SolverModel] = None,
        reward_fn: Optional[RubricsReward] = None,
    ):
        self.config = merge_config(config)
        self.cfg = self.config

        grpo_cfg = self.cfg.get("grpo", {})
        self.group_size = grpo_cfg.get("group_size", 4)
        self.clip_eps = grpo_cfg.get("clip_eps", 0.2)
        self.kl_beta = grpo_cfg.get("kl_beta", 0.04)
        self.adv_eps = grpo_cfg.get("adv_eps", 1e-8)
        self.max_grad_norm = grpo_cfg.get("max_grad_norm", 1.0)
        self.mu_iterations = grpo_cfg.get("mu_iterations", 1)

        self.challenge_model = challenge_model or self._create_challenge_model()
        self.solver_model = solver_model or self._create_solver_model()
        self.reward_fn = reward_fn or self._create_reward()

        self.ref_model = self._create_reference_model()

        train_cfg = self.cfg.get("training", {})
        lr = train_cfg.get("lr", 1e-5)
        self.optimizer = torch.optim.Adam(
            self.solver_model.model.parameters(), lr=lr
        )

        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        rw_cfg = self.cfg.get("reward", {})
        self.repetition_batch_size = rw_cfg.get("repetition_batch_size", 16)

    # -------------------------------------------------------------------
    # Model creation helpers
    # -------------------------------------------------------------------

    def _create_challenge_model(self) -> ChallengeModel:
        cm = self.cfg.get("challenge_model", {})
        return ChallengeModel(
            model_name=cm.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
            device=cm.get("device"),
        )

    def _create_solver_model(self) -> SolverModel:
        sm = self.cfg.get("solver_model", {})
        return SolverModel(
            model_name=sm.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
            device=sm.get("device"),
        )

    def _create_reference_model(self) -> SolverModel:
        """Create a frozen copy of the solver model as the KL reference."""
        sm = self.cfg.get("solver_model", {})
        ref = SolverModel(
            model_name=sm.get("model_name", "Qwen/Qwen2.5-1.5B-Instruct"),
            device=sm.get("device"),
        )
        ref.model.load_state_dict(self.solver_model.model.state_dict())
        ref.model.eval()
        for p in ref.model.parameters():
            p.requires_grad = False
        return ref

    def _create_reward(self) -> RubricsReward:
        from ..rewards.rubrics_reward import DynamicWeightScheduler, build_judge_api_client

        rw = self.cfg.get("reward", {})
        train_cfg = self.cfg.get("training", {})

        scheduler = None
        if rw.get("use_dynamic_weights", False):
            total_steps = max(
                train_cfg.get("epochs", 1) * self.cfg.get("data", {}).get("max_samples", 100),
                1,
            )
            scheduler = DynamicWeightScheduler(
                total_steps=total_steps,
                w1_init=rw.get("w1_init", 0.3),
                w1_final=rw.get("w1_final", 1.0),
                w2=rw.get("w2_repetition", 0.3),
                w3_init=rw.get("w3_init", 0.5),
                w3_final=rw.get("w3_final", 0.1),
                w4=rw.get("w4_relevance", 0.3),
                w5=rw.get("w5_rubric", 0.2),
            )

        use_llm = rw.get("use_llm_judge", True)
        client = build_judge_api_client() if use_llm else None

        return RubricsReward(
            use_llm_judge=use_llm,
            judge_model=rw.get("judge_model", "gpt-4o"),
            judge_temperature=rw.get("judge_temperature", 0.1),
            api_client=client,
            w1_adversarial=rw.get("w1_adversarial", 1.0),
            w2_repetition=rw.get("w2_repetition", 0.3),
            w3_format=rw.get("w3_format", 0.2),
            w4_relevance=rw.get("w4_relevance", 0.3),
            w5_rubric=rw.get("w5_rubric", 0.2),
            bleu_distance_threshold=rw.get("bleu_distance_threshold", 0.5),
            weight_scheduler=scheduler,
        )

    def _create_dataloader(self) -> CLBenchDataLoader:
        d = self.cfg.get("data", {})
        return CLBenchDataLoader(
            split=d.get("split", "train"),
            max_samples=d.get("max_samples"),
            subset=d.get("subset"),
            cache_dir=d.get("cache_dir"),
        )

    # -------------------------------------------------------------------
    # Core GRPO methods
    # -------------------------------------------------------------------

    def _sample_group(
        self,
        messages: List[dict],
    ) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        """Sample G responses from current policy for a single prompt."""
        sm_cfg = self.cfg.get("solver_model", {})
        return self.solver_model.generate_group(
            messages,
            group_size=self.group_size,
            max_new_tokens=sm_cfg.get("max_new_tokens", 2048),
            temperature=sm_cfg.get("temperature", 0.7),
        )

    def _compute_group_rewards(
        self,
        responses: List[str],
        rubrics: List[str],
        context: str,
        metadata: Dict[str, Any],
        **kwargs,
    ) -> Tuple[List[float], List[SolverRewardResult]]:
        """Compute rewards for each response in the group."""
        rewards = []
        breakdowns = []
        for resp in responses:
            result = self.reward_fn.compute_solver_reward(
                answer=resp,
                rubrics=rubrics,
                context=context,
                metadata=metadata,
                **kwargs,
            )
            rewards.append(result.total)
            breakdowns.append(result)
        return rewards, breakdowns

    @staticmethod
    def _compute_advantages(
        rewards: List[float],
        eps: float = 1e-8,
    ) -> List[float]:
        """
        Group-relative advantage normalization.
        A_i = (r_i - mean(r)) / (std(r) + eps)
        """
        t = torch.tensor(rewards, dtype=torch.float32)
        mean = t.mean()
        std = t.std()
        advantages = ((t - mean) / (std + eps)).tolist()
        return advantages

    def _grpo_loss_for_group(
        self,
        group: GRPOGroupResult,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss for one prompt's group of responses.

        Returns:
            (scalar loss tensor, dict of metrics for logging)
        """
        device = self.solver_model.model.device
        policy_losses = []
        kl_penalties = []
        ratios_list = []

        for i in range(len(group.responses)):
            gen_ids = group.generated_ids_list[i]
            if gen_ids.numel() == 0:
                continue

            input_ids = group.input_ids

            # Current policy per-token log-probs (differentiable)
            cur_logprobs = self.solver_model.compute_per_token_logprobs(
                input_ids, gen_ids
            )

            # Old policy per-token log-probs (from sampling time, detached)
            old_logprobs = self.solver_model.compute_per_token_logprobs_detached(
                input_ids, gen_ids
            )

            # Reference model per-token log-probs (frozen)
            ref_logprobs = self.ref_model.compute_per_token_logprobs_detached(
                input_ids, gen_ids
            )

            # Importance ratio: exp(cur - old)
            log_ratio = cur_logprobs - old_logprobs
            ratio = torch.exp(log_ratio)

            advantage = torch.tensor(
                group.advantages[i], device=device, dtype=torch.float32
            )

            # Clipped surrogate objective
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            policy_losses.append(policy_loss)

            # KL divergence: approx KL(pi_theta || pi_ref) per token
            kl = (cur_logprobs - ref_logprobs).mean()
            kl_penalties.append(kl)

            ratios_list.append(ratio.mean().item())

        if not policy_losses:
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        mean_policy_loss = torch.stack(policy_losses).mean()
        mean_kl = torch.stack(kl_penalties).mean()
        total_loss = mean_policy_loss + self.kl_beta * mean_kl

        metrics = {
            "policy_loss": mean_policy_loss.item(),
            "kl_penalty": mean_kl.item(),
            "total_loss": total_loss.item(),
            "mean_ratio": sum(ratios_list) / len(ratios_list) if ratios_list else 0.0,
            "mean_reward": sum(group.rewards) / len(group.rewards) if group.rewards else 0.0,
            "reward_std": torch.tensor(group.rewards).std().item() if len(group.rewards) > 1 else 0.0,
        }
        return total_loss, metrics

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------

    def train(self) -> Dict[str, float]:
        """
        Run GRPO training loop.

        For each prompt:
            1. Sample G responses from current policy
            2. Compute rewards for each response
            3. Compute group-relative advantages
            4. Run mu_iterations of policy gradient updates
            5. Log and checkpoint
        """
        loader = self._create_dataloader()
        loader.load()

        train_cfg = self.cfg.get("training", {})
        epochs = train_cfg.get("epochs", 1)
        log_every = train_cfg.get("log_every", 5)
        save_every = train_cfg.get("save_every", 100)

        samples = list(loader)
        global_step = 0

        total_loss = 0.0
        total_policy_loss = 0.0
        total_kl = 0.0
        total_reward = 0.0
        total_correctness = 0.0
        n_updates = 0

        for epoch in range(epochs):
            iterator = tqdm(samples, desc=f"GRPO Epoch {epoch + 1}/{epochs}")

            for sample in iterator:
                messages = sample.get("messages", [])
                rubrics = sample.get("rubrics", [])
                metadata = sample.get("metadata", {})

                if not messages:
                    continue

                # --- Step 1: Sample G responses ---
                group_outputs = self._sample_group(messages)
                responses = [r[0] for r in group_outputs]
                input_ids = group_outputs[0][1]
                generated_ids_list = [r[2] for r in group_outputs]

                # --- Step 2: Compute rewards ---
                context_str = self._extract_context(messages)
                rewards, breakdowns = self._compute_group_rewards(
                    responses, rubrics, context_str, metadata,
                )

                # --- Step 3: Compute group-relative advantages ---
                advantages = self._compute_advantages(rewards, eps=self.adv_eps)

                group = GRPOGroupResult(
                    prompt_messages=messages,
                    responses=responses,
                    rewards=rewards,
                    advantages=advantages,
                    input_ids=input_ids,
                    generated_ids_list=generated_ids_list,
                    reward_breakdowns=breakdowns,
                )

                # --- Step 4: Policy gradient update (mu iterations) ---
                for _ in range(self.mu_iterations):
                    self.solver_model.model.train()
                    self.optimizer.zero_grad()

                    loss, step_metrics = self._grpo_loss_for_group(group)
                    if loss.requires_grad:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.solver_model.model.parameters(),
                            self.max_grad_norm,
                        )
                        self.optimizer.step()

                global_step += 1
                n_updates += 1

                if step_metrics:
                    total_loss += step_metrics.get("total_loss", 0.0)
                    total_policy_loss += step_metrics.get("policy_loss", 0.0)
                    total_kl += step_metrics.get("kl_penalty", 0.0)
                    total_reward += step_metrics.get("mean_reward", 0.0)

                avg_correctness = 0.0
                if breakdowns:
                    avg_correctness = sum(
                        b.correctness for b in breakdowns
                    ) / len(breakdowns)
                total_correctness += avg_correctness

                # --- Logging ---
                if global_step % log_every == 0 and n_updates > 0:
                    iterator.set_postfix(
                        loss=round(total_loss / n_updates, 4),
                        p_loss=round(total_policy_loss / n_updates, 4),
                        kl=round(total_kl / n_updates, 4),
                        reward=round(total_reward / n_updates, 4),
                        correct=round(total_correctness / n_updates, 4),
                    )

                # --- Checkpoint ---
                if save_every and global_step % save_every == 0:
                    self._save_checkpoint(global_step)

        final_metrics = {
            "mean_total_loss": total_loss / n_updates if n_updates else 0.0,
            "mean_policy_loss": total_policy_loss / n_updates if n_updates else 0.0,
            "mean_kl_penalty": total_kl / n_updates if n_updates else 0.0,
            "mean_reward": total_reward / n_updates if n_updates else 0.0,
            "mean_correctness": total_correctness / n_updates if n_updates else 0.0,
            "global_steps": global_step,
            "epochs": epochs,
        }
        logger.info("GRPO training complete. Metrics: %s", final_metrics)

        self._save_checkpoint(global_step)
        return final_metrics

    # -------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------

    def _extract_context(self, messages: List[dict]) -> str:
        """Extract context string from messages for reward computation."""
        parts = []
        for m in messages:
            if isinstance(m, dict) and m.get("content"):
                parts.append(m["content"])
        return "\n".join(parts)

    def _save_checkpoint(self, step: int) -> None:
        """Save solver model checkpoint."""
        path = self.checkpoint_dir / f"grpo_solver_step_{step}"
        self.solver_model.model.save_pretrained(path)
        self.solver_model.tokenizer.save_pretrained(path)
        logger.info("Checkpoint saved at step %d -> %s", step, path)

    def sync_reference_model(self) -> None:
        """
        Sync reference model weights from the current solver model.
        Call periodically if you want the KL target to track training progress.
        """
        self.ref_model.model.load_state_dict(
            self.solver_model.model.state_dict()
        )
        logger.info("Reference model synced with current solver.")
