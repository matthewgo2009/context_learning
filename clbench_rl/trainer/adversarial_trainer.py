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

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from ..config.default_config import merge_config
from ..data.loader import CLBenchDataLoader
from ..models.challenge_model import ChallengeModel, parse_challenger_output
from ..models.solver_model import SolverModel
from ..rewards.base_reward import ChallengeRewardResult
from ..rewards.rubrics_reward import DynamicWeightScheduler, RubricsReward
from ..utils.metrics_logger import MetricsLogger

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

        self.solver_device, self.challenger_device = self._assign_devices()

        self.solver = self._build_solver()
        self.challenger = self._build_challenger()

        self.solver_ref = self._build_reference(self.solver)
        self.challenger_ref = self._build_reference_challenger(self.challenger)

        # Keep ref models on CPU to save ~8 GB per GPU;
        # they are moved to GPU briefly inside _do_grpo_update.
        self.solver_ref.model.to("cpu")
        self.challenger_ref.model.to("cpu")
        torch.cuda.empty_cache()
        logger.info("Reference models offloaded to CPU")

        self.reward_fn = self._build_reward()

        train_cfg = self.cfg.get("training", {})
        use_8bit = train_cfg.get("use_8bit_optimizer", True)
        AdamWCls = self._resolve_adamw(use_8bit)

        self.solver_optimizer = AdamWCls(
            self.solver.model.parameters(),
            lr=train_cfg.get("solver_lr", train_cfg.get("lr", 1e-5)),
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )
        self.challenger_optimizer = AdamWCls(
            self.challenger.model.parameters(),
            lr=train_cfg.get("challenger_lr", train_cfg.get("lr", 1e-5)),
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )

        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        rw_cfg = self.cfg.get("reward", {})
        self.rep_batch_size = rw_cfg.get("repetition_batch_size", 16)

    # ------------------------------------------------------------------
    # Device assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_adamw(use_8bit: bool):
        """Return an AdamW-like optimizer class.

        Prefer bitsandbytes PagedAdamW8bit when available — cuts optimizer
        state memory from ~32 GB to ~4 GB per 4B model, which is critical
        for fitting dual-model adversarial GRPO on 2×80 GB.
        """
        if use_8bit:
            try:
                import bitsandbytes as bnb
                logger.info("Using bitsandbytes PagedAdamW8bit (saves ~28 GB/GPU)")
                return bnb.optim.PagedAdamW8bit
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed — falling back to torch.optim.AdamW. "
                    "Install with: pip install bitsandbytes"
                )
        return torch.optim.AdamW

    @staticmethod
    def _assign_devices():
        n = torch.cuda.device_count()
        if n >= 2:
            logger.info("2+ GPUs detected — solver→cuda:0, challenger→cuda:1")
            return "cuda:0", "cuda:1"
        if n == 1:
            logger.info("Single GPU — all models on cuda:0")
            return "cuda:0", "cuda:0"
        return "cpu", "cpu"

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------

    def _build_solver(self) -> SolverModel:
        sm = self.cfg.get("solver_model", {})
        return SolverModel(
            model_name=sm.get("model_name", "Qwen/Qwen3-4B-Instruct-2507"),
            device=sm.get("device") or self.solver_device,
        )

    def _build_challenger(self) -> ChallengeModel:
        cm = self.cfg.get("challenge_model", {})
        return ChallengeModel(
            model_name=cm.get("model_name", "Qwen/Qwen3-4B-Instruct-2507"),
            device=cm.get("device") or self.challenger_device,
        )

    def _build_reference(self, solver: SolverModel) -> SolverModel:
        sm = self.cfg.get("solver_model", {})
        ref = SolverModel(
            model_name=sm.get("model_name", "Qwen/Qwen3-4B-Instruct-2507"),
            device=sm.get("device") or self.solver_device,
        )
        ref.model.load_state_dict(solver.model.state_dict())
        ref.model.eval()
        for p in ref.model.parameters():
            p.requires_grad = False
        return ref

    def _build_reference_challenger(self, challenger: ChallengeModel) -> ChallengeModel:
        cm = self.cfg.get("challenge_model", {})
        ref = ChallengeModel(
            model_name=cm.get("model_name", "Qwen/Qwen3-4B-Instruct-2507"),
            device=cm.get("device") or self.challenger_device,
        )
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
            judge_model=rw.get("judge_model", "gpt-4o-mini"),
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

    def _do_grpo_update(self, model, ref_model, optimizer, group: GroupResult):
        """Per-trajectory GRPO with ref-model CPU offloading.

        Phase 1: bring ref model to GPU, compute ref logprobs, send it back.
        Phase 2: per-trajectory forward + immediate backward (no gradient
                 checkpointing needed — only 1 graph alive at a time, and
                 ref is off GPU so there's ~8 GB extra room).
        """
        metrics: Dict[str, float] = {}
        device = model.model.device
        G = len(group.responses)
        input_ids = group.input_ids.to(device)

        # ------ Phase 1: ref logprobs (ref briefly on GPU) ------
        ref_model.model.to(device)
        ref_lps: List[Optional[torch.Tensor]] = []
        for i in range(G):
            gen_ids = group.generated_ids_list[i].to(device)
            if gen_ids.numel() == 0:
                ref_lps.append(None)
                continue
            lp = ref_model.compute_per_token_logprobs_detached(input_ids, gen_ids)
            ref_lps.append(lp.cpu())
        ref_model.model.to("cpu")
        torch.cuda.empty_cache()

        # ------ Phase 2: GRPO mu-iterations (ref off GPU) ------
        for _ in range(self.mu_iterations):
            model.model.train()
            optimizer.zero_grad()
            n_valid = 0
            sum_policy = 0.0
            sum_kl = 0.0

            for i in range(G):
                gen_ids = group.generated_ids_list[i].to(device)
                if gen_ids.numel() == 0 or ref_lps[i] is None:
                    continue

                cur_lp = model.compute_per_token_logprobs(input_ids, gen_ids)
                old_lp = cur_lp.detach()

                ratio = torch.exp(cur_lp - old_lp)
                adv = torch.tensor(
                    group.advantages[i], device=device, dtype=torch.float32
                )
                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                ) * adv
                p_loss_i = -torch.min(surr1, surr2).mean()
                kl_i = (cur_lp - ref_lps[i].to(device)).mean()

                loss_i = (p_loss_i + self.kl_beta * kl_i) / max(G, 1)
                loss_i.backward()

                sum_policy += p_loss_i.item()
                sum_kl += kl_i.item()
                n_valid += 1

            if n_valid > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.model.parameters(), self.max_grad_norm
                )
                optimizer.step()

            n = max(n_valid, 1)
            metrics = {
                "policy_loss": sum_policy / n,
                "kl": sum_kl / n,
                "total_loss": (sum_policy + self.kl_beta * sum_kl) / n,
            }

        model.model.eval()
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

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"

    @staticmethod
    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): AdversarialTrainer._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [AdversarialTrainer._json_safe(x) for x in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    def _append_rollout_trace(
        self,
        path: Path,
        record: Dict[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

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

        save_rollouts = train_cfg.get("save_rollout_traces", True)
        is_rank0 = int(os.environ.get("LOCAL_RANK", "0")) == 0
        ckpt_root = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        trace_dir_cfg = train_cfg.get("rollout_trace_dir")
        rollout_trace_dir = (
            Path(trace_dir_cfg)
            if trace_dir_cfg
            else ckpt_root / "rollout_traces"
        )
        max_ctx = int(train_cfg.get("rollout_log_max_context_chars", 8000))
        max_field = int(train_cfg.get("rollout_log_max_field_chars", 16000))
        rollout_jsonl = rollout_trace_dir / "training_rollouts.jsonl"
        if save_rollouts and is_rank0:
            rollout_trace_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Rollout traces → %s", rollout_jsonl)

        cm_cfg = self.cfg.get("challenge_model", {})
        sm_cfg = self.cfg.get("solver_model", {})

        # Input-length caps to bound activation memory.
        max_ctx_ch = int(train_cfg.get("max_context_chars_challenger", 6000))
        max_ctx_sv = int(train_cfg.get("max_context_chars_solver", 4000))

        samples = list(loader)
        global_step = 0

        acc = {
            "solver_reward": 0.0, "challenger_reward": 0.0,
            "j_score": 0.0, "r_adv": 0.0, "r_rep": 0.0,
            "r_fmt": 0.0, "r_rel": 0.0, "r_rubric": 0.0,
            "solver_loss": 0.0, "challenger_loss": 0.0,
        }
        n = 0

        metrics_path = Path(train_cfg.get("checkpoint_dir", "checkpoints")) / "metrics.jsonl"
        ml = MetricsLogger(metrics_path, flush_every=max(1, log_every))

        for epoch in range(epochs):
            iterator = tqdm(samples, desc=f"Epoch {epoch+1}/{epochs}")

            batch_questions: List[str] = []

            for sample in iterator:
                messages = sample.get("messages", [])
                if not messages:
                    continue

                context = self._extract_context(messages)
                ch_context = context[:max_ctx_ch] if max_ctx_ch > 0 else context
                sv_context = context[:max_ctx_sv] if max_ctx_sv > 0 else context
                challenger_messages = self.challenger.build_context_messages(ch_context)

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
                c_reward_details: List[ChallengeRewardResult] = []
                best_idx = 0
                best_c_reward = float("-inf")

                for gi, parsed in enumerate(parsed_outputs):
                    q_text = parsed.question or c_texts[gi]
                    solver_msgs = [
                        {"role": "system", "content": sv_context[:2000]},
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
                    c_reward_details.append(c_result)

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
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # === Phase 3: Solver generates G answers for best (Q,E,R) ===
                best_parsed = parsed_outputs[best_idx]
                best_q = best_parsed.question or c_texts[best_idx]
                best_rubrics = [best_parsed.rubric] if best_parsed.rubric else sample.get("rubrics", [])

                solver_prompt = [
                    {"role": "system", "content": sv_context},
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
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # === Metrics ===
                self.reward_fn.step()
                global_step += 1
                n += 1

                if save_rollouts and is_rank0:
                    cand = []
                    for gi, parsed in enumerate(parsed_outputs):
                        cr = c_reward_details[gi]
                        cand.append({
                            "index": gi,
                            "question": self._truncate_text(
                                parsed.question or "", max_field
                            ),
                            "evidence_span": self._truncate_text(
                                parsed.evidence_span or "", max_field
                            ),
                            "rubric": self._truncate_text(
                                parsed.rubric or "", max_field
                            ),
                            "raw_response": self._truncate_text(
                                c_texts[gi], max_field
                            ),
                            "total_reward": cr.total,
                            "adversarial": cr.adversarial,
                            "repetition_penalty": cr.repetition_penalty,
                            "format_penalty": cr.format_penalty,
                            "relevance": cr.relevance,
                            "rubric_quality": cr.rubric_quality,
                            "details": self._json_safe(cr.details),
                        })
                    solver_rows = []
                    for j, (ans, sr) in enumerate(zip(s_texts, s_breakdowns)):
                        solver_rows.append({
                            "index": j,
                            "answer": self._truncate_text(ans, max_field),
                            "total_reward": sr.total,
                            "correctness": sr.correctness,
                            "details": self._json_safe(sr.details),
                        })
                    trace_rec = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "sample_metadata": self._json_safe(
                            sample.get("metadata") or {}
                        ),
                        "context_preview": self._truncate_text(context, max_ctx),
                        "challenger_group_size": len(c_texts),
                        "best_challenger_index": best_idx,
                        "mean_challenger_reward": sum(c_rewards) / max(len(c_rewards), 1),
                        "challenger_candidates": cand,
                        "best_question": self._truncate_text(best_q, max_field),
                        "best_rubric": self._truncate_text(
                            best_parsed.rubric or "", max_field
                        ),
                        "solver_on_best": {
                            "mean_reward": sum(s_rewards) / max(len(s_rewards), 1),
                            "mean_judge_score": sum(
                                sr.correctness for sr in s_breakdowns
                            ) / max(len(s_breakdowns), 1),
                            "answers": solver_rows,
                        },
                        "challenger_loss": c_metrics.get("total_loss", 0.0),
                        "solver_loss": s_metrics.get("total_loss", 0.0),
                    }
                    self._append_rollout_trace(rollout_jsonl, trace_rec)

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
                    ml.log(
                        step=global_step, epoch=epoch,
                        j_score=mean_j,
                        solver_reward=sum(s_rewards) / max(len(s_rewards), 1),
                        challenger_reward=sum(c_rewards) / max(len(c_rewards), 1),
                        solver_loss=s_metrics.get("total_loss", 0.0),
                        challenger_loss=c_metrics.get("total_loss", 0.0),
                        solver_policy_loss=s_metrics.get("policy_loss", 0.0),
                        challenger_policy_loss=c_metrics.get("policy_loss", 0.0),
                        solver_kl=s_metrics.get("kl_penalty", 0.0),
                        challenger_kl=c_metrics.get("kl_penalty", 0.0),
                        avg_j_score=acc["j_score"] / n,
                        avg_solver_reward=acc["solver_reward"] / n,
                        avg_challenger_reward=acc["challenger_reward"] / n,
                    )

                if save_every and global_step % save_every == 0:
                    self._save_checkpoint(global_step)

                if ref_sync_every and global_step % ref_sync_every == 0:
                    self._sync_references()

            # Trim batch_questions per epoch to avoid unbounded growth
            batch_questions = batch_questions[-self.rep_batch_size * 4:]

        self._save_checkpoint(global_step)
        ml.close()
        logger.info("Metrics saved to %s", metrics_path)

        out = {k: v / max(n, 1) for k, v in acc.items()} | {
            "global_steps": global_step,
            "epochs": epochs,
            "metrics_file": str(metrics_path),
        }
        if save_rollouts and is_rank0:
            out["rollout_trace_file"] = str(rollout_jsonl)
        return out

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
        cpu_sd = {k: v.cpu() for k, v in self.solver.model.state_dict().items()}
        self.solver_ref.model.load_state_dict(cpu_sd)
        cpu_sd = {k: v.cpu() for k, v in self.challenger.model.state_dict().items()}
        self.challenger_ref.model.load_state_dict(cpu_sd)
        logger.info("Reference models synced (on CPU).")
