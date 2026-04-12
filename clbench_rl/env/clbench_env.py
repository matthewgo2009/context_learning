"""RL environment encapsulating sample -> Challenge -> Solver -> Scoring.

Aligned with the Self-Evolving ICL framework:
    1. Challenger generates (Q, E, R) from context C.
    2. Solver generates answer A from (C, Q).
    3. Judge scores A against R.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..models.challenge_model import ChallengeModel, parse_challenger_output
from ..models.solver_model import SolverModel
from ..rewards.base_reward import BaseReward, ChallengeRewardResult, SolverRewardResult


@dataclass
class EnvStep:
    """Output of a single environment step."""

    solver_reward: float
    challenge_reward: float
    solver_reward_breakdown: Optional[SolverRewardResult] = None
    challenge_reward_breakdown: Optional[ChallengeRewardResult] = None
    answer: str = ""
    challenge_output: Optional[str] = None
    evidence_span: str = ""
    messages: List[dict] = field(default_factory=list)
    rubrics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CLBenchEnv:
    """
    Environment that runs: sample -> (optional Challenge transform) -> Solver generate -> score.
    """

    def __init__(
        self,
        challenge_model: ChallengeModel,
        solver_model: SolverModel,
        reward_fn: BaseReward,
        challenge_pass_through: bool = True,
    ):
        self.challenge_model = challenge_model
        self.solver_model = solver_model
        self.reward_fn = reward_fn
        self.challenge_pass_through = challenge_pass_through

    def step(
        self,
        sample: Dict[str, Any],
        return_logprobs: bool = False,
        batch_repetition_penalty: Optional[float] = None,
    ) -> EnvStep:
        """
        Run one episode: get messages from sample -> Solver answers -> compute rewards.

        Args:
            sample: Dict with 'messages', 'rubrics', 'metadata'.
            return_logprobs: If True, Solver returns (answer, logprobs) for RL training.
            batch_repetition_penalty: Pre-computed BLEU-clustering penalty (B.4)
                from the trainer. If provided, overrides single-text penalty.

        Returns:
            EnvStep with rewards, answer, and context. logprobs stored in metadata if requested.
        """
        messages = sample.get("messages", [])
        rubrics = sample.get("rubrics", [])
        metadata = sample.get("metadata", {})

        if not messages:
            return EnvStep(
                solver_reward=0.0,
                challenge_reward=0.0,
                answer="",
                challenge_output=None,
                messages=messages,
                rubrics=rubrics,
                metadata=metadata,
            )

        challenge_output = None
        evidence_span = ""

        if self.challenge_pass_through:
            solver_messages = messages
        else:
            solver_messages = self.challenge_model.format_prompt(messages)
            raw_output = self.challenge_model.generate(messages)
            challenge_output = raw_output

            if raw_output:
                parsed = parse_challenger_output(raw_output)
                evidence_span = parsed.evidence_span

                q_text = parsed.question or raw_output
                solver_messages = [
                    {"role": "system", "content": messages[0].get("content", "") if messages else ""},
                    {"role": "user", "content": q_text},
                ]

                if parsed.rubric:
                    rubrics = [parsed.rubric]

        if return_logprobs:
            answer, logprobs = self.solver_model.generate(
                solver_messages, return_logprobs=True
            )
            metadata = {**metadata, "_logprobs": logprobs}
        else:
            answer = self.solver_model.generate(solver_messages)

        context_str = self._extract_context(messages)
        question_str = self._extract_question(solver_messages)

        # --- Solver reward: R_s = J_score(A, R) ---
        solver_result = self.reward_fn.compute_solver_reward(
            answer=answer,
            rubrics=rubrics,
            context=context_str,
            metadata=metadata,
        )

        # --- Challenge reward: R_c = w1·R_adv - w2·R_rep - ... ---
        challenge_kwargs: Dict[str, Any] = {}
        if batch_repetition_penalty is not None:
            challenge_kwargs["batch_repetition_penalty"] = batch_repetition_penalty

        challenge_result = self.reward_fn.compute_challenge_reward(
            solver_correctness=solver_result.correctness,
            challenge_output=challenge_output or "",
            context=context_str,
            question=question_str,
            rubrics=rubrics,
            metadata=metadata,
            evidence_span=evidence_span,
            **challenge_kwargs,
        )

        return EnvStep(
            solver_reward=solver_result.total,
            challenge_reward=challenge_result.total,
            solver_reward_breakdown=solver_result,
            challenge_reward_breakdown=challenge_result,
            answer=answer,
            challenge_output=challenge_output,
            evidence_span=evidence_span,
            messages=messages,
            rubrics=rubrics,
            metadata=metadata,
        )

    def _extract_context(self, messages: List[dict]) -> str:
        """Extract context string from messages for reward computation."""
        parts = []
        for m in messages:
            if isinstance(m, dict) and m.get("content"):
                parts.append(m["content"])
        return "\n".join(parts)

    def _extract_question(self, messages: List[dict]) -> str:
        """Extract the user question from the last user message."""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                return m.get("content", "")
        return ""
