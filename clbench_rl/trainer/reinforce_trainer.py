"""REINFORCE-style trainer for Solver (and optionally Challenge) model.

Implements batch-level BLEU-clustering repetition penalty (Appendix B.4).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ..data.loader import CLBenchDataLoader
from ..env.clbench_env import CLBenchEnv
from ..models.challenge_model import ChallengeModel
from ..models.solver_model import SolverModel
from ..rewards.rubrics_reward import RubricsReward

logger = logging.getLogger(__name__)


class ReinforceTrainer:
    """
    Trainer that runs pipeline: load data -> env.step -> accumulate rewards -> update.

    Uses REINFORCE for Solver (and optionally Challenge) with policy gradient.
    Applies batch-level BLEU-clustering repetition penalty per Appendix B.4.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        challenge_model: Optional[ChallengeModel] = None,
        solver_model: Optional[SolverModel] = None,
        reward_fn: Optional[RubricsReward] = None,
    ):
        from ..config.default_config import merge_config

        self.config = merge_config(config)
        self.cfg = self.config

        self.challenge_model = challenge_model or self._create_challenge_model()
        self.solver_model = solver_model or self._create_solver_model()
        self.reward_fn = reward_fn or self._create_reward()

        self.env = CLBenchEnv(
            challenge_model=self.challenge_model,
            solver_model=self.solver_model,
            reward_fn=self.reward_fn,
            challenge_pass_through=True,
        )

        self.checkpoint_dir = Path(
            self.cfg.get("training", {}).get("checkpoint_dir", "checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        rw_cfg = self.cfg.get("reward", {})
        self.repetition_batch_size = rw_cfg.get("repetition_batch_size", 16)

    def _create_challenge_model(self) -> ChallengeModel:
        cm = self.cfg.get("challenge_model", {})
        return ChallengeModel(
            model_name=cm.get("model_name", "Qwen/Qwen3-4B-Instruct-2507"),
            device=cm.get("device"),
        )

    def _create_solver_model(self) -> SolverModel:
        sm = self.cfg.get("solver_model", {})
        return SolverModel(
            model_name=sm.get("model_name", "Qwen/Qwen3-4B-Instruct-2507"),
            device=sm.get("device"),
        )

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
        loader = CLBenchDataLoader(
            split=d.get("split", "train"),
            max_samples=d.get("max_samples"),
            subset=d.get("subset"),
            cache_dir=d.get("cache_dir"),
        )
        return loader

    def run_episode(
        self,
        sample: Dict[str, Any],
        return_logprobs: bool = False,
        batch_repetition_penalty: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run one episode and return step result with rewards."""
        step = self.env.step(
            sample,
            return_logprobs=return_logprobs,
            batch_repetition_penalty=batch_repetition_penalty,
        )
        out = {
            "solver_reward": step.solver_reward,
            "challenge_reward": step.challenge_reward,
            "solver_breakdown": step.solver_reward_breakdown,
            "challenge_breakdown": step.challenge_reward_breakdown,
            "answer": step.answer,
            "challenge_output": step.challenge_output,
            "metadata": step.metadata,
        }
        if "_logprobs" in step.metadata:
            out["logprobs"] = step.metadata.pop("_logprobs", None)
        return out

    def _extract_question_from_sample(self, sample: Dict[str, Any]) -> str:
        """Extract the question text from a sample for batch repetition computation."""
        messages = sample.get("messages", [])
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                return m.get("content", "")
        return ""

    def train(self) -> Dict[str, float]:
        """
        Run training loop with batch-level BLEU repetition penalty (B.4).

        Processes samples in batches of `repetition_batch_size`. For each batch:
            1. Extract questions from all samples
            2. Compute BLEU-clustering penalties: r_rep(x_i) = |C_k| / B
            3. Run episodes with per-sample repetition penalties injected
        """
        loader = self._create_dataloader()
        loader.load()

        train_cfg = self.cfg.get("training", {})
        log_every = train_cfg.get("log_every", 5)
        save_every = train_cfg.get("save_every", 100)

        total_solver_r = 0.0
        total_challenge_r = 0.0
        total_j_score = 0.0
        total_r_adv = 0.0
        total_r_rep = 0.0
        total_r_fmt = 0.0
        total_r_rel = 0.0
        total_r_rubric = 0.0
        n = 0
        samples = list(loader)

        batches = self._chunk_samples(samples, self.repetition_batch_size)
        flat_idx = 0
        iterator = tqdm(total=len(samples), desc="Training")

        for batch in batches:
            questions = [self._extract_question_from_sample(s) for s in batch]
            rep_penalties = self.reward_fn.compute_batch_repetition(questions)

            for j, sample in enumerate(batch):
                result = self.run_episode(
                    sample,
                    return_logprobs=False,
                    batch_repetition_penalty=rep_penalties[j],
                )
                r_s = result["solver_reward"]
                r_c = result["challenge_reward"]
                total_solver_r += r_s
                total_challenge_r += r_c
                n += 1

                s_bd = result.get("solver_breakdown")
                c_bd = result.get("challenge_breakdown")
                if s_bd:
                    total_j_score += s_bd.correctness
                if c_bd:
                    total_r_adv += c_bd.adversarial
                    total_r_rep += c_bd.repetition_penalty
                    total_r_fmt += c_bd.format_penalty
                    total_r_rel += c_bd.relevance
                    total_r_rubric += c_bd.rubric_quality

                self.reward_fn.step()
                flat_idx += 1
                iterator.update(1)

                if flat_idx % log_every == 0:
                    iterator.set_postfix(
                        j_score=round(total_j_score / n, 4),
                        r_c=round(total_challenge_r / n, 4),
                        r_adv=round(total_r_adv / n, 4),
                        r_rep=round(total_r_rep / n, 4),
                    )

                if save_every and flat_idx % save_every == 0:
                    self._save_checkpoint(flat_idx)

        iterator.close()

        metrics = {
            "mean_solver_reward": total_solver_r / n if n else 0.0,
            "mean_challenge_reward": total_challenge_r / n if n else 0.0,
            "mean_j_score": total_j_score / n if n else 0.0,
            "mean_r_adv": total_r_adv / n if n else 0.0,
            "mean_r_rep": total_r_rep / n if n else 0.0,
            "mean_r_fmt": total_r_fmt / n if n else 0.0,
            "mean_r_rel": total_r_rel / n if n else 0.0,
            "mean_r_rubric": total_r_rubric / n if n else 0.0,
            "num_episodes": n,
        }
        logger.info("Training complete. Metrics: %s", metrics)
        return metrics

    @staticmethod
    def _chunk_samples(
        samples: List[Dict[str, Any]],
        batch_size: int,
    ) -> List[List[Dict[str, Any]]]:
        """Split samples into batches for batch-level repetition penalty."""
        return [
            samples[i : i + batch_size]
            for i in range(0, len(samples), batch_size)
        ]

    def _save_checkpoint(self, step: int) -> None:
        """Save model checkpoints."""
        train_cfg = self.cfg.get("training", {})
        if train_cfg.get("train_solver"):
            path = self.checkpoint_dir / f"solver_step_{step}"
            self.solver_model.model.save_pretrained(path)
            self.solver_model.tokenizer.save_pretrained(path)
        if train_cfg.get("train_challenge"):
            path = self.checkpoint_dir / f"challenge_step_{step}"
            self.challenge_model.model.save_pretrained(path)
            self.challenge_model.tokenizer.save_pretrained(path)
