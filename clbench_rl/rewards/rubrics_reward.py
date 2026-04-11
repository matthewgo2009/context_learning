"""Rubrics-based reward aligned with the Self-Evolving ICL framework.

Implements the asymmetric adversarial play reward design:
  Challenger reward: w1*R_adv - w2*R_rep - w3*R_fmt + w4*R_rel + w5*R_rubric
  Solver reward:     J_score(A, R) with context grounding and tool-usage bonuses

Key components:
  - R_adv   = 1 - J_score(A, R)
  - R_rep   = |C_k| / B  (BLEU-clustering batch penalty, Appendix B.4)
  - R_fmt   = binary format penalty
  - R_rel   = BLEU(Q, C) · I[J_ans(Q, C) = True]
  - R_rubric= Similarity(R, E) · J_align(R, E, C)

Judge: GPT-4o as frozen evaluator (Appendix B.3)
"""

import json
import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base_reward import BaseReward, ChallengeRewardResult, SolverRewardResult

logger = logging.getLogger(__name__)

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False
    logger.warning("nltk not available; BLEU-based penalties will use n-gram fallback.")

try:
    from sklearn.cluster import AgglomerativeClustering

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available; repetition penalty clustering will use n-gram fallback.")


def build_rubrics_text(rubrics: List) -> str:
    """Build rubrics checklist from rubrics list."""
    if not rubrics:
        return "No specific rubrics provided."
    lines = []
    for i, rubric in enumerate(rubrics, 1):
        if isinstance(rubric, dict):
            criteria = rubric.get("rubric_criteria", rubric.get("criteria", "")).strip()
        else:
            criteria = str(rubric).strip()
        if criteria:
            lines.append(f"{i}. {criteria}")
    return "\n".join(lines) if lines else "No specific rubrics provided."


# ---------------------------------------------------------------------------
# Repetition penalty  R_rep  (Appendix B.4)
#
# 1. Pairwise distance: d_ij = 1 - BLEU(x_i, x_j)
#    using nltk sentence_bleu with SmoothingFunction().method1
# 2. Agglomerative clustering: metric='precomputed', linkage='average'
# 3. Penalty: r_rep(x_i) = |C_k| / B
# ---------------------------------------------------------------------------

def _bleu_distance(x_i: str, x_j: str) -> float:
    """Compute BLEU-based distance: d_ij = 1 - BLEU(x_i, x_j)."""
    ref = x_i.split()
    hyp = x_j.split()
    if not ref or not hyp:
        return 1.0
    smoothing = SmoothingFunction().method1
    score = sentence_bleu([ref], hyp, smoothing_function=smoothing)
    return 1.0 - score


def compute_bleu_distance_matrix(questions: List[str]) -> np.ndarray:
    """
    Compute pairwise BLEU-based distance matrix for a batch of questions.

    d_ij = 1 - BLEU(x_i, x_j), tokenized by whitespace splitting,
    smoothed with SmoothingFunction().method1.
    """
    n = len(questions)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _bleu_distance(questions[i], questions[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def compute_batch_repetition_penalties(
    questions: List[str],
    distance_threshold: float = 0.5,
) -> List[float]:
    """
    Compute per-question repetition penalty for a batch using BLEU clustering.

    Algorithm (Appendix B.4):
        1. Compute pairwise distance: d_ij = 1 - BLEU(x_i, x_j)
        2. Agglomerative clustering (metric='precomputed', linkage='average')
        3. r_rep(x_i) = |C_k| / B, where C_k is the cluster of x_i, B = batch size

    Penalty semantics: a value near 1/B means the question is unique in the batch;
    a value near 1.0 means the entire batch is one cluster (all similar).

    Args:
        questions: List of question strings in the batch.
        distance_threshold: Clustering distance threshold for agglomerative clustering.

    Returns:
        List of penalty values, one per question. Higher = more repetition.
    """
    n = len(questions)
    if n <= 1:
        return [0.0] * n

    if not (_NLTK_AVAILABLE and _SKLEARN_AVAILABLE):
        return _ngram_fallback_batch_penalties(questions)

    dist_matrix = compute_bleu_distance_matrix(questions)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = clustering.fit_predict(dist_matrix)

    cluster_sizes = Counter(labels)
    penalties = [cluster_sizes[labels[i]] / n for i in range(n)]
    return penalties


def _ngram_fallback_batch_penalties(questions: List[str]) -> List[float]:
    """Fallback when nltk/sklearn are not available: token-overlap based similarity."""
    n = len(questions)
    if n <= 1:
        return [0.0] * n

    token_sets = [set(q.lower().split()) for q in questions]
    penalties = []
    for i in range(n):
        sim_count = 0
        for j in range(n):
            if i == j:
                continue
            if not token_sets[i] or not token_sets[j]:
                continue
            overlap = len(token_sets[i] & token_sets[j])
            union = len(token_sets[i] | token_sets[j])
            if union > 0 and overlap / union > 0.5:
                sim_count += 1
        penalties.append((sim_count + 1) / n)
    return penalties


# ---------------------------------------------------------------------------
# Single-text repetition (used when batch info is unavailable)
# ---------------------------------------------------------------------------

def _extract_ngrams(text: str, n: int) -> List[str]:
    """Extract word-level n-grams from text."""
    tokens = text.split()
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_single_repetition_penalty(
    text: str,
    ngram_sizes: Tuple[int, ...] = (3, 4, 5),
    max_penalty: float = 0.5,
) -> float:
    """
    Fallback single-text repetition penalty based on repeated n-gram ratio.
    Used only when batch-level BLEU penalty is not applicable.
    Returns a value in [-max_penalty, 0.0].
    """
    if not text or not text.strip():
        return 0.0

    total_ratio = 0.0
    count = 0
    for n in ngram_sizes:
        ngrams = _extract_ngrams(text, n)
        if not ngrams:
            continue
        freq = Counter(ngrams)
        repeated = sum(c - 1 for c in freq.values() if c > 1)
        ratio = repeated / len(ngrams)
        total_ratio += ratio
        count += 1

    if count == 0:
        return 0.0
    avg_ratio = total_ratio / count
    return -max_penalty * min(avg_ratio * 3.0, 1.0)


# ---------------------------------------------------------------------------
# Format check  R_fmt  (binary penalty for structural constraint violations)
# ---------------------------------------------------------------------------

def compute_format_penalty(
    text: str,
    min_length: int = 20,
    max_penalty: float = 1.0,
) -> float:
    """
    Binary format penalty R_fmt.

    Checks: minimum length, presence of structured sections (<question>,
    <evidence>, <rubric> tags), and absence of degenerate patterns.
    Returns 0.0 (no violation) or -1.0 (violation detected).
    """
    if not text or not text.strip():
        return -max_penalty

    stripped = text.strip()
    violations = 0
    total_checks = 4

    if len(stripped) < min_length:
        violations += 1

    has_question = "<question>" in stripped and "</question>" in stripped
    if not has_question:
        violations += 1

    has_evidence = "<evidence>" in stripped and "</evidence>" in stripped
    if not has_evidence:
        violations += 1

    alpha_ratio = sum(c.isalpha() or c.isspace() for c in stripped) / max(len(stripped), 1)
    if alpha_ratio < 0.3:
        violations += 1

    if violations > 0:
        return -max_penalty * min(violations / total_checks, 1.0)
    return 0.0


# ---------------------------------------------------------------------------
# Context relevance  R_rel = BLEU(Q, C) · I[J_ans(Q, C) = True]
# ---------------------------------------------------------------------------

def compute_bleu_context_relevance(question: str, context: str) -> float:
    """
    Compute BLEU(Q, C) — lexical overlap between question and context.

    The context is treated as the reference and the question as the hypothesis.
    Returns a float in [0.0, 1.0].
    """
    if not question or not context:
        return 0.0

    q_tokens = question.lower().split()
    c_tokens = context.lower().split()

    if not q_tokens or not c_tokens:
        return 0.0

    if _NLTK_AVAILABLE:
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(
            [c_tokens], q_tokens,
            weights=(0.5, 0.3, 0.15, 0.05),
            smoothing_function=smoothing,
        )
        return float(score)

    q_set = set(q_tokens)
    c_set = set(c_tokens)
    if not q_set:
        return 0.0
    return len(q_set & c_set) / len(q_set)


def compute_context_question_relevance_heuristic(
    context: str,
    question: str,
) -> float:
    """
    Heuristic measure of context-question relevance via token overlap.
    Returns a value in [0.0, 1.0]. Used as fallback when LLM judge unavailable.
    """
    if not context or not question:
        return 0.0

    ctx_tokens = set(context.lower().split())
    q_tokens = set(question.lower().split())

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "but", "with", "this", "that",
        "it", "be", "as", "by", "from", "not", "do", "does", "did",
        "what", "which", "who", "how", "when", "where", "why",
    }
    ctx_tokens -= stop_words
    q_tokens -= stop_words

    if not q_tokens:
        return 0.0

    overlap = ctx_tokens & q_tokens
    return len(overlap) / len(q_tokens)


# ---------------------------------------------------------------------------
# Evidence-Rubric similarity  Similarity(R, E)
# ---------------------------------------------------------------------------

def compute_evidence_rubric_similarity(
    rubrics: List[str],
    evidence_span: str,
) -> float:
    """
    Compute Similarity(R, E) — how well rubrics are anchored to evidence span.

    Uses BLEU if available, otherwise falls back to token overlap.
    Returns a float in [0.0, 1.0].
    """
    if not rubrics or not evidence_span or not evidence_span.strip():
        return 0.0

    rubric_text = " ".join(str(r).strip() for r in rubrics if str(r).strip())
    if not rubric_text:
        return 0.0

    r_tokens = rubric_text.lower().split()
    e_tokens = evidence_span.lower().split()

    if not r_tokens or not e_tokens:
        return 0.0

    if _NLTK_AVAILABLE:
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(
            [e_tokens], r_tokens,
            weights=(0.4, 0.3, 0.2, 0.1),
            smoothing_function=smoothing,
        )
        return float(score)

    r_set = set(r_tokens)
    e_set = set(e_tokens)
    if not r_set:
        return 0.0
    return len(r_set & e_set) / len(r_set)


# ---------------------------------------------------------------------------
# Context grounding (Solver answer grounded in context, not pretrained knowledge)
# ---------------------------------------------------------------------------

def compute_context_grounding_heuristic(
    answer: str,
    context: str,
    threshold: float = 0.15,
) -> float:
    """
    Heuristic check whether the solver's answer is grounded in the context
    rather than hallucinated from pretrained knowledge.

    Measures overlap of answer content tokens with context tokens.
    Returns 1.0 if well-grounded, scaled down toward 0.0 otherwise.
    """
    if not answer or not context:
        return 0.0

    ans_tokens = set(answer.lower().split())
    ctx_tokens = set(context.lower().split())

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "but", "with", "this", "that",
        "it", "be", "as", "by", "from", "not", "i", "my", "your",
    }
    ans_tokens -= stop_words
    ctx_tokens -= stop_words

    if not ans_tokens:
        return 0.0

    overlap = ans_tokens & ctx_tokens
    ratio = len(overlap) / len(ans_tokens)

    if ratio >= threshold:
        return min(ratio / 0.5, 1.0)
    return ratio / threshold * 0.5


# ---------------------------------------------------------------------------
# Tool usage detection (Solver referencing or using context via tools)
# ---------------------------------------------------------------------------

TOOL_USE_PATTERNS = [
    r"\b(according to|based on|from the|the (context|passage|text|document) (says|states|mentions|indicates|shows))",
    r"\b(as (stated|mentioned|described|shown) in)",
    r"\b(referring to|reference to|cited from)",
    r"\b(the (given|provided) (information|data|context|text))",
    r"<tool_call>.*?</tool_call>",
    r"\[tool\].*?\[/tool\]",
    r"```tool\b",
]


def detect_tool_usage(answer: str) -> float:
    """
    Detect if the solver's answer shows evidence of tool usage or explicit
    context referencing. Returns a score in [0.0, 1.0].
    """
    if not answer:
        return 0.0

    matches = 0
    for pattern in TOOL_USE_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE | re.DOTALL):
            matches += 1

    return min(matches / 3.0, 1.0)


# ---------------------------------------------------------------------------
# Dynamic weight scheduler for w1..w5
# ---------------------------------------------------------------------------

class DynamicWeightScheduler:
    """Dynamic hyperparameter scheduler for challenger reward weights w1..w5.

    During early training, validity constraints (R_fmt, R_rel, R_rubric) are
    emphasized. As training progresses, the adversarial signal (R_adv) is
    ramped up to increase difficulty.

    Schedule:
        - w1 (adversarial): ramps from w1_init to w1_final via cosine schedule
        - w2 (repetition):  constant
        - w3 (format):      decays from w3_init to w3_final
        - w4 (relevance):   constant
        - w5 (rubric):      constant
    """

    def __init__(
        self,
        total_steps: int,
        w1_init: float = 0.3,
        w1_final: float = 1.0,
        w2: float = 0.3,
        w3_init: float = 0.5,
        w3_final: float = 0.1,
        w4: float = 0.3,
        w5: float = 0.2,
    ):
        self.total_steps = max(total_steps, 1)
        self.w1_init = w1_init
        self.w1_final = w1_final
        self.w2 = w2
        self.w3_init = w3_init
        self.w3_final = w3_final
        self.w4 = w4
        self.w5 = w5

    def get_weights(self, step: int) -> Dict[str, float]:
        """Return current weights for the given training step."""
        t = min(step / self.total_steps, 1.0)

        cosine_ramp = 0.5 * (1.0 - math.cos(math.pi * t))
        w1 = self.w1_init + (self.w1_final - self.w1_init) * cosine_ramp
        w3 = self.w3_init + (self.w3_final - self.w3_init) * cosine_ramp

        return {
            "w1_adversarial": w1,
            "w2_repetition": self.w2,
            "w3_format": w3,
            "w4_relevance": self.w4,
            "w5_rubric": self.w5,
        }


# ===========================================================================
# Main reward class
# ===========================================================================

class RubricsReward(BaseReward):
    """
    Multi-component reward aligned with the Self-Evolving ICL framework.

    Challenger reward (5 components from the paper):
        R_c = w1·R_adv - w2·R_rep - w3·R_fmt + w4·R_rel + w5·R_rubric
        where:
            R_adv   = 1 - J_score(A, R)
            R_rep   = |C_k| / B  (BLEU clustering)
            R_fmt   = binary format penalty
            R_rel   = BLEU(Q, C) · I[J_ans(Q, C) = True]
            R_rubric= Similarity(R, E) · J_align(R, E, C)

    Solver reward:
        R_s = J_score(A, R) + grounding_bonus + tool_usage_bonus
    """

    def __init__(
        self,
        use_llm_judge: bool = False,
        judge_model: str = "gpt-4o",
        judge_temperature: float = 0.1,
        api_client=None,
        sub_category_weights: Optional[Dict[str, float]] = None,
        # Challenger reward weights (default / static)
        w1_adversarial: float = 1.0,
        w2_repetition: float = 0.3,
        w3_format: float = 0.2,
        w4_relevance: float = 0.3,
        w5_rubric: float = 0.2,
        # Solver reward weights
        solver_correctness_weight: float = 1.0,
        context_grounding_weight: float = 0.3,
        tool_usage_weight: float = 0.2,
        # Repetition penalty clustering params
        bleu_distance_threshold: float = 0.5,
        # Optional dynamic weight scheduler
        weight_scheduler: Optional[DynamicWeightScheduler] = None,
    ):
        self.use_llm_judge = use_llm_judge
        self.judge_model = judge_model
        self.judge_temperature = judge_temperature
        self.api_client = api_client
        self.sub_category_weights = sub_category_weights or {}

        self.w1_adversarial = w1_adversarial
        self.w2_repetition = w2_repetition
        self.w3_format = w3_format
        self.w4_relevance = w4_relevance
        self.w5_rubric = w5_rubric

        self.solver_correctness_weight = solver_correctness_weight
        self.context_grounding_weight = context_grounding_weight
        self.tool_usage_weight = tool_usage_weight

        self.bleu_distance_threshold = bleu_distance_threshold
        self.weight_scheduler = weight_scheduler
        self._current_step = 0

    def _get_challenge_weights(self) -> Dict[str, float]:
        """Get current challenger weights, using scheduler if available."""
        if self.weight_scheduler is not None:
            return self.weight_scheduler.get_weights(self._current_step)
        return {
            "w1_adversarial": self.w1_adversarial,
            "w2_repetition": self.w2_repetition,
            "w3_format": self.w3_format,
            "w4_relevance": self.w4_relevance,
            "w5_rubric": self.w5_rubric,
        }

    def step(self) -> None:
        """Advance training step counter for dynamic weight scheduling."""
        self._current_step += 1

    # -----------------------------------------------------------------------
    # Solver Reward
    # -----------------------------------------------------------------------

    def compute_solver_reward(
        self,
        answer: str,
        rubrics: List[str],
        context: str,
        metadata: Dict[str, Any],
        **kwargs,
    ) -> SolverRewardResult:
        """
        Compute solver reward: maximize J_score(A, R).

        Components:
            1. correctness      : J_score(A, R) from Judge LLM or heuristic
            2. context_grounding: bonus for context-derived answers
            3. tool_usage       : bonus for explicit context referencing
        """
        result = SolverRewardResult()

        if not answer or not answer.strip():
            return result

        ground_truth = kwargs.get("ground_truth", "")

        # --- 1. J_score(A, R) via GPT-4o judge (B.3) ---
        if self.use_llm_judge and self.api_client:
            if ground_truth:
                correctness = self._gpt4o_judge_correctness(answer, ground_truth)
            else:
                correctness = self._llm_judge_correctness_rubrics(answer, rubrics)
        else:
            correctness = self._heuristic_grade(answer, rubrics)
        result.correctness = correctness

        # --- 2. Context grounding reward ---
        grounding = compute_context_grounding_heuristic(answer, context)
        if self.use_llm_judge and self.api_client:
            llm_grounding = self._llm_judge_context_grounding(answer, context)
            grounding = 0.5 * grounding + 0.5 * llm_grounding
        result.context_grounding = grounding

        # --- 3. Tool usage / context referencing reward ---
        result.tool_usage = detect_tool_usage(answer)

        # --- Weighted total ---
        sub = metadata.get("sub_category", "")
        cat_weight = self.sub_category_weights.get(sub, 1.0)

        result.total = cat_weight * (
            self.solver_correctness_weight * result.correctness
            + self.context_grounding_weight * result.context_grounding
            + self.tool_usage_weight * result.tool_usage
        )

        result.details = {
            "correctness_raw": correctness,
            "grounding_raw": grounding,
            "tool_usage_raw": result.tool_usage,
            "sub_category": sub,
            "cat_weight": cat_weight,
            "judge_mode": "gpt4o" if (self.use_llm_judge and self.api_client) else "heuristic",
        }
        return result

    def _heuristic_grade(self, answer: str, rubrics: List) -> float:
        """Rule-based fallback: keyword matching against rubrics."""
        if not answer or len(answer.strip()) < 10:
            return 0.0

        valid_rubrics = [str(r).strip().lower() for r in rubrics if str(r).strip()]
        if not valid_rubrics:
            return 1.0 if len(answer) > 50 else 0.5

        answer_lower = answer.lower()
        matched = sum(
            1 for r in valid_rubrics
            if any(kw in answer_lower for kw in r.split() if len(kw) > 3)
        )
        return min(matched / max(len(valid_rubrics), 1), 1.0)

    def _gpt4o_judge_correctness(self, answer: str, ground_truth: str) -> float:
        """
        GPT-4o as judge for answer correctness (Appendix B.3).
        Returns: 1.0 if "Yes", 0.0 if "No" or error.
        """
        try:
            response = self.api_client.chat.completions.create(
                model=self.judge_model,
                temperature=self.judge_temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an answer correctness checker.",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Hi, there is an answer: {answer},\n"
                            f"and the ground truth answer is: {ground_truth},\n"
                            "please check whether the answer is correct or not, "
                            "and return the **only** Yes or No."
                        ),
                    },
                ],
            )
            text = response.choices[0].message.content.strip().lower()
            if "yes" in text:
                return 1.0
            return 0.0
        except Exception as e:
            logger.warning("GPT-4o judge correctness call failed: %s", e)
            return 0.0

    def _llm_judge_correctness_rubrics(self, answer: str, rubrics: List) -> float:
        """J_score(A, R): evaluate answer against rubrics."""
        rubrics_text = build_rubrics_text(rubrics)
        try:
            response = self.api_client.chat.completions.create(
                model=self.judge_model,
                temperature=self.judge_temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict grading judge.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Evaluate the student answer against the rubrics.\n"
                            "Output ONLY a JSON object: "
                            '{"score": 0 or 1, "rationale": "brief explanation"}\n'
                            "Score 1 ONLY if ALL rubric requirements are fully satisfied.\n\n"
                            f"Rubrics:\n{rubrics_text}\n\n"
                            f"Student answer:\n{answer}\n\n"
                            "JSON output:"
                        ),
                    },
                ],
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            data = json.loads(text)
            return float(data.get("score", 0))
        except Exception as e:
            logger.warning("LLM judge rubrics call failed: %s", e)
            return 0.0

    def _llm_judge_context_grounding(self, answer: str, context: str) -> float:
        """Verify the answer is derived from context, not pretrained knowledge."""
        try:
            response = self.api_client.chat.completions.create(
                model=self.judge_model,
                temperature=self.judge_temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a grounding verification judge.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Determine whether the answer is derived from the given "
                            "context or appears to be hallucinated / based on external "
                            "knowledge.\n"
                            "Output ONLY a JSON object: "
                            '{"grounded": 0 or 1, "rationale": "brief explanation"}\n'
                            "Score 1 if the answer is clearly supported by the context.\n\n"
                            f"Context:\n{context[:3000]}\n\n"
                            f"Answer:\n{answer}\n\n"
                            "JSON output:"
                        ),
                    },
                ],
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            data = json.loads(text)
            return float(data.get("grounded", 0))
        except Exception as e:
            logger.warning("LLM judge grounding call failed: %s", e)
            return 0.0

    # -----------------------------------------------------------------------
    # Challenge Reward
    # -----------------------------------------------------------------------

    def compute_challenge_reward(
        self,
        solver_correctness: float,
        challenge_output: str,
        context: str,
        question: str,
        rubrics: List[str],
        metadata: Dict[str, Any],
        evidence_span: str = "",
        **kwargs,
    ) -> ChallengeRewardResult:
        """
        Compute challenger reward:
            R_c = w1·R_adv - w2·R_rep - w3·R_fmt + w4·R_rel + w5·R_rubric

        Components:
            1. R_adv   = 1 - J_score(A, R)              adversarial
            2. R_rep   = |C_k|/B or single-text fallback repetition penalty
            3. R_fmt   = binary format penalty            format check
            4. R_rel   = BLEU(Q,C) · I[J_ans(Q,C)=True]  context relevance
            5. R_rubric= Sim(R,E) · J_align(R,E,C)       rubric quality
        """
        result = ChallengeRewardResult()
        output_text = challenge_output or ""
        weights = self._get_challenge_weights()

        # --- 1. R_adv = 1 - J_score(A, R) ---
        result.adversarial = 1.0 - solver_correctness

        # --- 2. R_rep ---
        batch_rep = kwargs.get("batch_repetition_penalty")
        if batch_rep is not None:
            result.repetition_penalty = float(batch_rep)
        else:
            result.repetition_penalty = abs(compute_single_repetition_penalty(output_text))

        # --- 3. R_fmt (binary) ---
        result.format_penalty = abs(compute_format_penalty(output_text))

        # --- 4. R_rel = BLEU(Q, C) · I[J_ans(Q, C) = True] ---
        bleu_qc = compute_bleu_context_relevance(question, context)
        if self.use_llm_judge and self.api_client:
            answerability = self._llm_judge_answerability(question, context)
        else:
            answerability = self._heuristic_answerability(question, context)
        result.relevance = bleu_qc * answerability

        # --- 5. R_rubric = Similarity(R, E) · J_align(R, E, C) ---
        sim_re = compute_evidence_rubric_similarity(rubrics, evidence_span)
        if self.use_llm_judge and self.api_client and evidence_span:
            alignment = self._llm_judge_rubric_evidence_alignment(
                rubrics, evidence_span, context
            )
        elif evidence_span:
            alignment = self._heuristic_rubric_evidence_alignment(
                rubrics, evidence_span, context
            )
        else:
            alignment = self._heuristic_rubric_quality(rubrics) if not evidence_span else 0.0
        result.rubric_quality = sim_re * alignment if evidence_span else alignment

        # --- Weighted total: w1·R_adv - w2·R_rep - w3·R_fmt + w4·R_rel + w5·R_rubric ---
        w1 = weights["w1_adversarial"]
        w2 = weights["w2_repetition"]
        w3 = weights["w3_format"]
        w4 = weights["w4_relevance"]
        w5 = weights["w5_rubric"]

        result.total = (
            w1 * result.adversarial
            - w2 * result.repetition_penalty
            - w3 * result.format_penalty
            + w4 * result.relevance
            + w5 * result.rubric_quality
        )

        result.details = {
            "solver_correctness_input": solver_correctness,
            "r_adv": result.adversarial,
            "r_rep": result.repetition_penalty,
            "r_fmt": result.format_penalty,
            "r_rel": result.relevance,
            "r_rubric": result.rubric_quality,
            "bleu_qc": bleu_qc,
            "answerability": answerability,
            "sim_re": sim_re,
            "alignment": alignment,
            "evidence_span_provided": bool(evidence_span),
            "weights": weights,
            "used_batch_repetition": batch_rep is not None,
            "training_step": self._current_step,
        }
        return result

    # -----------------------------------------------------------------------
    # Answerability check  I[J_ans(Q, C) = True]
    # -----------------------------------------------------------------------

    def _llm_judge_answerability(self, question: str, context: str) -> float:
        """
        I[J_ans(Q, C) = True]: indicator function via Judge LLM.
        Returns 1.0 if the question is answerable strictly from context, else 0.0.
        """
        try:
            response = self.api_client.chat.completions.create(
                model=self.judge_model,
                temperature=self.judge_temperature,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an answerability judge. Determine whether a question "
                            "can be answered STRICTLY from the provided context, without "
                            "requiring any external knowledge."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Can the following question be answered strictly using "
                            "ONLY the information in the provided context?\n\n"
                            f"Context:\n{context[:3000]}\n\n"
                            f"Question:\n{question}\n\n"
                            "Output ONLY a JSON object: "
                            '{"answerable": true or false, "rationale": "brief explanation"}\n'
                            "JSON output:"
                        ),
                    },
                ],
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            data = json.loads(text)
            return 1.0 if data.get("answerable", False) else 0.0
        except Exception as e:
            logger.warning("LLM judge answerability call failed: %s", e)
            return 0.0

    def _heuristic_answerability(self, question: str, context: str) -> float:
        """
        Heuristic approximation of I[J_ans(Q, C)].
        Uses content-word overlap as a proxy: if enough question keywords
        appear in the context, we consider the question answerable.
        """
        relevance = compute_context_question_relevance_heuristic(context, question)
        return 1.0 if relevance >= 0.3 else 0.0

    # -----------------------------------------------------------------------
    # Rubric-Evidence alignment  J_align(R, E, C)
    # -----------------------------------------------------------------------

    def _llm_judge_rubric_evidence_alignment(
        self,
        rubrics: List[str],
        evidence_span: str,
        context: str,
    ) -> float:
        """
        J_align(R, E, C): Judge verifies that rubric R is faithfully aligned
        with evidence span E within context C.
        Returns a score in [0.0, 1.0].
        """
        rubrics_text = build_rubrics_text(rubrics)
        try:
            response = self.api_client.chat.completions.create(
                model=self.judge_model,
                temperature=self.judge_temperature,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a rubric alignment judge. Evaluate whether the "
                            "rubric criteria are faithfully anchored to the provided "
                            "evidence span extracted from the context."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Rate the alignment between the rubric and the evidence span.\n"
                            "The rubric should be based on information from the evidence span, "
                            "and the evidence span should come from the context.\n\n"
                            "Output ONLY a JSON object: "
                            '{"alignment": <float 0.0-1.0>, "rationale": "brief explanation"}\n'
                            "1.0 = rubric is perfectly grounded in the evidence span.\n"
                            "0.0 = rubric is unrelated to the evidence.\n\n"
                            f"Context (excerpt):\n{context[:2000]}\n\n"
                            f"Evidence Span:\n{evidence_span[:1000]}\n\n"
                            f"Rubric:\n{rubrics_text}\n\n"
                            "JSON output:"
                        ),
                    },
                ],
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            data = json.loads(text)
            raw = float(data.get("alignment", 0))
            return max(0.0, min(1.0, raw))
        except Exception as e:
            logger.warning("LLM judge rubric-evidence alignment call failed: %s", e)
            return 0.0

    def _heuristic_rubric_evidence_alignment(
        self,
        rubrics: List[str],
        evidence_span: str,
        context: str,
    ) -> float:
        """
        Heuristic approximation of J_align(R, E, C).
        Checks that evidence span exists in context and rubric tokens overlap with evidence.
        """
        if not evidence_span or not context:
            return 0.0

        e_in_c = evidence_span.strip().lower() in context.lower()
        if not e_in_c:
            e_tokens = set(evidence_span.lower().split())
            c_tokens = set(context.lower().split())
            e_in_c = len(e_tokens & c_tokens) / max(len(e_tokens), 1) > 0.7

        rubric_text = " ".join(str(r).strip() for r in rubrics if str(r).strip())
        r_tokens = set(rubric_text.lower().split())
        e_tokens = set(evidence_span.lower().split())

        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "but", "with", "this", "that",
        }
        r_tokens -= stop_words
        e_tokens -= stop_words

        if not r_tokens or not e_tokens:
            return 0.5 if e_in_c else 0.0

        overlap = len(r_tokens & e_tokens) / len(r_tokens)

        if e_in_c:
            return min(overlap * 1.2, 1.0)
        return overlap * 0.5

    # -----------------------------------------------------------------------
    # Relevance (LLM fallback, kept for backward compatibility)
    # -----------------------------------------------------------------------

    def _llm_judge_relevance(self, context: str, question: str) -> float:
        """LLM-based relevance — used as part of R_rel when answerability check passes."""
        try:
            response = self.api_client.chat.completions.create(
                model=self.judge_model,
                temperature=self.judge_temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are evaluating question quality.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Rate how relevant and meaningful the question is "
                            "with respect to the given context.\n"
                            "Output ONLY a JSON object: "
                            '{"score": <float 0.0-1.0>, "rationale": "brief explanation"}\n'
                            "1.0 = highly relevant, well-formed question.\n"
                            "0.0 = irrelevant or nonsensical.\n\n"
                            f"Context:\n{context[:3000]}\n\n"
                            f"Question:\n{question}\n\n"
                            "JSON output:"
                        ),
                    },
                ],
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            data = json.loads(text)
            raw = float(data.get("score", 0))
            return max(0.0, min(1.0, raw))
        except Exception as e:
            logger.warning("LLM judge relevance call failed: %s", e)
            return 0.0

    # -----------------------------------------------------------------------
    # Rubric quality (standalone, used when evidence_span is not provided)
    # -----------------------------------------------------------------------

    def _heuristic_rubric_quality(self, rubrics: List) -> float:
        """
        Heuristic evaluation of rubric quality.
        Good rubrics: multiple criteria, each with sufficient detail.
        """
        valid = [str(r).strip() for r in rubrics if str(r).strip()]
        if not valid:
            return 0.0

        count_score = min(len(valid) / 3.0, 1.0)

        avg_len = sum(len(r) for r in valid) / len(valid)
        detail_score = min(avg_len / 50.0, 1.0)

        diversity = len(set(r.lower() for r in valid)) / len(valid)

        return (count_score + detail_score + diversity) / 3.0

    def _llm_judge_rubric_quality(self, question: str, rubrics: List) -> float:
        """LLM-based rubric quality evaluation (without evidence span)."""
        rubrics_text = build_rubrics_text(rubrics)
        try:
            response = self.api_client.chat.completions.create(
                model=self.judge_model,
                temperature=self.judge_temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are evaluating rubric quality.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Rate how well the rubrics serve as evaluation criteria "
                            "for the given question.\n"
                            "Output ONLY a JSON object: "
                            '{"score": <float 0.0-1.0>, "rationale": "brief explanation"}\n'
                            "1.0 = comprehensive, specific, well-matched rubrics.\n"
                            "0.0 = missing, vague, or irrelevant rubrics.\n\n"
                            f"Question:\n{question}\n\n"
                            f"Rubrics:\n{rubrics_text}\n\n"
                            "JSON output:"
                        ),
                    },
                ],
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            data = json.loads(text)
            raw = float(data.get("score", 0))
            return max(0.0, min(1.0, raw))
        except Exception as e:
            logger.warning("LLM judge rubric quality call failed: %s", e)
            return 0.0

    # -----------------------------------------------------------------------
    # Batch-level repetition (public API)
    # -----------------------------------------------------------------------

    def compute_batch_repetition(self, questions: List[str]) -> List[float]:
        """
        Compute batch-level repetition penalties (Appendix B.4).

        Algorithm:
            1. Pairwise distance: d_ij = 1 - BLEU(x_i, x_j)
            2. Agglomerative clustering (precomputed, average linkage)
            3. r_rep(x_i) = |C_k| / B

        Args:
            questions: All questions generated in the current batch.

        Returns:
            Per-question penalty values.
        """
        return compute_batch_repetition_penalties(
            questions,
            distance_threshold=self.bleu_distance_threshold,
        )
