"""Adversarial self-play trainer aligned with the Self-Evolving ICL paper.

Both Solver (πθs) and Challenger (πθc) are trained via GRPO:

  Solver objective:     max E[J_score(A, R)]
  Challenger objective: max E[w1·R_adv - w2·R_rep - w3·R_fmt + w4·R_rel + w5·R_rubric]

Training loop (per epoch):
  For each context C in the dataset:
    1. Challenger generates G candidate (Q,E,R) from C   → compute R_C for each
    2. Pick best (Q,E,R) by R_C; Solver generates G answers A  → compute J_score
    3. GRPO update on Challenger using challenger rewards
    4. GRPO update on Solver using solver rewards
    5. Advance dynamic weight scheduler

Supports DeepSpeed ZeRO via Accelerate for 8×H100 training.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from ..config.default_config import merge_config
from ..data.loader import CLBenchDataLoader
from ..models.challenge_model import ChallengeModel, parse_challenger_output
from ..models.solver_model import SolverModel
from ..rewards.base_reward import ChallengeRewardResult, SolverRewardResult
from ..rewards.rubrics_reward import DynamicWeightScheduler, RubricsReward

logger = logging.getLogger(__name__)


@dataclass
class GroupResult:
    """Result of one GRPO group (shared by Solver and Challenger)."""
    responses: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)
    input_ids: Optional[torch.Tensor] = None
    generated_ids_list: List[torch.Tensor] = field(default_factory=list)


class AdversarialTrainer:
    """Dual-model GRPO trainer: trains both Solver and Challenger.

    Paper: "The framework is trained using an actor-critic or direct evolution
    approach with the following objectives" — both θs and θc are optimized.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = merge_config(config)

        grpo = self.cfg.get("grpo", {})
        self.group_size = grpo.get("group_size", 4)
        self.clip_eps = grpo.get("clip_eps", 0.2)
        self.kl_beta = grpo.get("kl_beta", 0.04)
        self.adv_eps = grpo.get("adv_eps", 1e-8)
        self.max_grad_norm = grpo.get("max_grad_norm", 1.0)
        self.mu_iterations = grpo.get("mu_iterations", 1)

        self.solver = self._build_solver()
        self.challenger = self._build_challenger()
        self.solver_ref = self._build_reference(self.solver)
        self.challenger_ref = self._build_reference_challenger(self.challenger)
        self.reward_fn = self._build_reward()

        train_cfg = self.cfg.get("training", {})
        self.solver_optimizer = torch.optim.AdamW(
            self.solver.model.parameters(),
            lr=train_cfg.get("solver_lr", train_cfg.get("lr", 1e-5)),
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )
        self.challenger_optimizer = torch.optim.AdamW(
            self.challenger.model.parameters(),
            lr=train_cfg.get("challenger_lr", train_cfg.get("lr", 1e-5)),
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )

        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        rw_cfg = self.cfg.get("reward", {})
        self.rep_batch_size = rw_cfg.get("repetition_batch_size", 16)

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------

    def _build_solver(self) -> SolverModel:
        sm = self.cfg.get("solver_model", {})
        return SolverModel(
            model_name=sm.get("model_name", "Qwen/Qwen2.5-3B-Instruct"),
            device=sm.get("device"),
        )

    def _build_challenger(self) -> ChallengeModel:
        cm = self.cfg.get("challenge_model", {})
        return ChallengeModel(
            model_name=cm.get("model_name", "Qwen/Qwen2.5-3B-Instruct"),
            device=cm.get("device"),
        )

    def _build_reference(self, solver: SolverModel) -> SolverModel:
        sm = self.cfg.get("solver_model", {})
        ref = SolverModel(model_name=sm.get("model_name", "Qwen/Qwen2.5-3B-Instruct"),
                          device=sm.get("device"))
        ref.model.load_state_dict(solver.model.state_dict())
        ref.model.eval()
        for p in ref.model.parameters():
            p.requires_grad = False
        return ref

    def _build_reference_challenger(self, challenger: ChallengeModel) -> ChallengeModel:
        cm = self.cfg.get("challenge_model", {})
        ref = ChallengeModel(model_name=cm.get("model_name", "Qwen/Qwen2.5-3B-Instruct"),
                             device=cm.get("device"))
        ref.model.load_state_dict(challenger.model.state_dict())
        ref.model.eval()
        for p in ref.model.parameters():
            p.requires_grad = False
        return ref

    def _build_reward(self) -> RubricsReward:
        rw = self.cfg.get("reward", {})
        train_cfg = self.cfg.get("training", {})

        scheduler = None
        if rw.get("use_dynamic_weights", True):
            data_cfg = self.cfg.get("data", {})
            total_steps = max(
                train_cfg.get("epochs", 3) * (data_cfg.get("max_samples") or 5000),
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

        from ..rewards.rubrics_reward import build_judge_api_client

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

    # ------------------------------------------------------------------
    # GRPO primitives (shared between Solver & Challenger)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_advantages(rewards: List[float], eps: float = 1e-8) -> List[float]:
        t = torch.tensor(rewards, dtype=torch.float32)
        return ((t - t.mean()) / (t.std() + eps)).tolist()

    def _grpo_loss(
        self,
        model,
        ref_model,
        group: GroupResult,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute clipped policy gradient loss + KL penalty for a group."""
        device = model.model.device
        policy_losses = []
        kl_penalties = []

        for i in range(len(group.responses)):
            gen_ids = group.generated_ids_list[i]
            if gen_ids.numel() == 0:
                continue

            cur_lp = model.compute_per_token_logprobs(group.input_ids, gen_ids)
            old_lp = model.compute_per_token_logprobs_detached(group.input_ids, gen_ids)
            ref_lp = ref_model.compute_per_token_logprobs_detached(group.input_ids, gen_ids)

            ratio = torch.exp(cur_lp - old_lp)
            adv = torch.tensor(group.advantages[i], device=device, dtype=torch.float32)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            policy_losses.append(-torch.min(surr1, surr2).mean())
            kl_penalties.append((cur_lp - ref_lp).mean())

        if not policy_losses:
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        p_loss = torch.stack(policy_losses).mean()
        kl = torch.stack(kl_penalties).mean()
        total = p_loss + self.kl_beta * kl

        return total, {
            "policy_loss": p_loss.item(),
            "kl": kl.item(),
            "total_loss": total.item(),
        }

    def _do_grpo_update(self, model, ref_model, optimizer, group: GroupResult):
        """Run mu_iterations of GRPO updates on `model`."""
        metrics = {}
        for _ in range(self.mu_iterations):
            model.model.train()
            optimizer.zero_grad()
            loss, metrics = self._grpo_loss(model, ref_model, group)
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.model.parameters(), self.max_grad_norm
                )
                optimizer.step()
        return metrics

    # ------------------------------------------------------------------
    # Context extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_context(messages: List[dict]) -> str:
        parts = []
        for m in messages:
            if isinstance(m, dict) and m.get("content"):
                parts.append(m["content"])
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, float]:
        """Run adversarial self-play training.

        Per-step:
          1. Challenger samples G outputs (Q,E,R) from context C
          2. Compute challenger rewards R_C for each
          3. GRPO update Challenger on R_C
          4. Pick best (Q,E,R); Solver samples G answers
          5. Compute solver rewards J_score for each
          6. GRPO update Solver on J_score
        """
        loader = CLBenchDataLoader(**{
            k: v for k, v in self.cfg.get("data", {}).items()
            if k in ("split", "max_samples", "subset", "cache_dir")
        })
        loader.load()

        train_cfg = self.cfg.get("training", {})
        epochs = train_cfg.get("epochs", 3)
        log_every = train_cfg.get("log_every", 10)
        save_every = train_cfg.get("save_every", 500)
        ref_sync_every = train_cfg.get("ref_sync_every", 200)

        cm_cfg = self.cfg.get("challenge_model", {})
        sm_cfg = self.cfg.get("solver_model", {})

        samples = list(loader)
        global_step = 0

        acc = {
            "solver_reward": 0.0, "challenger_reward": 0.0,
            "j_score": 0.0, "r_adv": 0.0, "r_rep": 0.0,
            "r_fmt": 0.0, "r_rel": 0.0, "r_rubric": 0.0,
            "solver_loss": 0.0, "challenger_loss": 0.0,
        }
        n = 0

        for epoch in range(epochs):
            iterator = tqdm(samples, desc=f"Epoch {epoch+1}/{epochs}")

            batch_questions: List[str] = []

            for sample in iterator:
                messages = sample.get("messages", [])
                if not messages:
                    continue

                context = self._extract_context(messages)
                challenger_messages = self.challenger.build_context_messages(context)

                # === Phase 1: Challenger generates G (Q,E,R) ===
                c_group_raw = self.challenger.generate_group(
                    challenger_messages,
                    group_size=self.group_size,
                    max_new_tokens=cm_cfg.get("max_new_tokens", 1024),
                    temperature=cm_cfg.get("temperature", 0.7),
                )

                c_texts = [r[0] for r in c_group_raw]
                c_input_ids = c_group_raw[0][1]
                c_gen_ids_list = [r[2] for r in c_group_raw]

                parsed_outputs = [parse_challenger_output(t) for t in c_texts]
                questions = [p.question or p.raw for p in parsed_outputs]
                batch_questions.extend(questions)

                # Compute batch-level repetition penalties
                if len(batch_questions) >= self.rep_batch_size:
                    rep_penalties = self.reward_fn.compute_batch_repetition(
                        batch_questions[-self.rep_batch_size:]
                    )
                else:
                    rep_penalties = self.reward_fn.compute_batch_repetition(batch_questions)

                # For each challenger output, compute challenger reward
                # We need a solver answer first — generate one quick answer per (Q,E,R)
                c_rewards = []
                best_idx = 0
                best_c_reward = float("-inf")

                for gi, parsed in enumerate(parsed_outputs):
                    q_text = parsed.question or c_texts[gi]
                    solver_msgs = [
                        {"role": "system", "content": context[:2000]},
                        {"role": "user", "content": q_text},
                    ]
                    with torch.no_grad():
                        answer = self.solver.generate(
                            solver_msgs,
                            max_new_tokens=sm_cfg.get("max_new_tokens", 2048),
                            temperature=0.3,
                        )

                    rubrics = [parsed.rubric] if parsed.rubric else sample.get("rubrics", [])

                    s_result = self.reward_fn.compute_solver_reward(
                        answer=answer, rubrics=rubrics,
                        context=context, metadata=sample.get("metadata", {}),
                    )

                    rep_idx = min(gi, len(rep_penalties) - 1) if rep_penalties else 0
                    rep_val = rep_penalties[rep_idx] if rep_penalties else 0.0

                    c_result = self.reward_fn.compute_challenge_reward(
                        solver_correctness=s_result.correctness,
                        challenge_output=c_texts[gi],
                        context=context,
                        question=parsed.question,
                        rubrics=rubrics,
                        metadata=sample.get("metadata", {}),
                        evidence_span=parsed.evidence_span,
                        batch_repetition_penalty=rep_val,
                    )
                    c_rewards.append(c_result.total)

                    if c_result.total > best_c_reward:
                        best_c_reward = c_result.total
                        best_idx = gi

                # === Phase 2: GRPO update Challenger ===
                c_advantages = self._compute_advantages(c_rewards, self.adv_eps)
                c_group = GroupResult(
                    responses=c_texts,
                    rewards=c_rewards,
                    advantages=c_advantages,
                    input_ids=c_input_ids,
                    generated_ids_list=c_gen_ids_list,
                )
                c_metrics = self._do_grpo_update(
                    self.challenger, self.challenger_ref,
                    self.challenger_optimizer, c_group,
                )

                # === Phase 3: Solver generates G answers for best (Q,E,R) ===
                best_parsed = parsed_outputs[best_idx]
                best_q = best_parsed.question or c_texts[best_idx]
                best_rubrics = [best_parsed.rubric] if best_parsed.rubric else sample.get("rubrics", [])

                solver_prompt = [
                    {"role": "system", "content": context[:4000]},
                    {"role": "user", "content": best_q},
                ]

                s_group_raw = self.solver.generate_group(
                    solver_prompt,
                    group_size=self.group_size,
                    max_new_tokens=sm_cfg.get("max_new_tokens", 2048),
                    temperature=sm_cfg.get("temperature", 0.7),
                )

                s_texts = [r[0] for r in s_group_raw]
                s_input_ids = s_group_raw[0][1]
                s_gen_ids_list = [r[2] for r in s_group_raw]

                s_rewards = []
                s_breakdowns = []
                for ans in s_texts:
                    sr = self.reward_fn.compute_solver_reward(
                        answer=ans, rubrics=best_rubrics,
                        context=context, metadata=sample.get("metadata", {}),
                    )
                    s_rewards.append(sr.total)
                    s_breakdowns.append(sr)

                # === Phase 4: GRPO update Solver ===
                s_advantages = self._compute_advantages(s_rewards, self.adv_eps)
                s_group = GroupResult(
                    responses=s_texts,
                    rewards=s_rewards,
                    advantages=s_advantages,
                    input_ids=s_input_ids,
                    generated_ids_list=s_gen_ids_list,
                )
                s_metrics = self._do_grpo_update(
                    self.solver, self.solver_ref,
                    self.solver_optimizer, s_group,
                )

                # === Metrics ===
                self.reward_fn.step()
                global_step += 1
                n += 1

                mean_j = sum(sr.correctness for sr in s_breakdowns) / max(len(s_breakdowns), 1)
                acc["solver_reward"] += sum(s_rewards) / max(len(s_rewards), 1)
                acc["challenger_reward"] += sum(c_rewards) / max(len(c_rewards), 1)
                acc["j_score"] += mean_j
                acc["solver_loss"] += s_metrics.get("total_loss", 0.0)
                acc["challenger_loss"] += c_metrics.get("total_loss", 0.0)

                if global_step % log_every == 0 and n > 0:
                    iterator.set_postfix(
                        j=round(acc["j_score"] / n, 4),
                        s_loss=round(acc["solver_loss"] / n, 4),
                        c_loss=round(acc["challenger_loss"] / n, 4),
                        c_r=round(acc["challenger_reward"] / n, 4),
                    )

                if save_every and global_step % save_every == 0:
                    self._save_checkpoint(global_step)

                if ref_sync_every and global_step % ref_sync_every == 0:
                    self._sync_references()

            # Trim batch_questions per epoch to avoid unbounded growth
            batch_questions = batch_questions[-self.rep_batch_size * 4:]

        self._save_checkpoint(global_step)

        return {k: v / max(n, 1) for k, v in acc.items()} | {
            "global_steps": global_step, "epochs": epochs,
        }

    # ------------------------------------------------------------------
    # Checkpointing & reference sync
    # ------------------------------------------------------------------

    def _save_checkpoint(self, step: int) -> None:
        for name, model in [("solver", self.solver), ("challenger", self.challenger)]:
            path = self.checkpoint_dir / f"{name}_step_{step}"
            model.model.save_pretrained(path)
            model.tokenizer.save_pretrained(path)
        logger.info("Checkpoint saved at step %d", step)

    def _sync_references(self) -> None:
        self.solver_ref.model.load_state_dict(self.solver.model.state_dict())
        self.challenger_ref.model.load_state_dict(self.challenger.model.state_dict())
        logger.info("Reference models synced.")
