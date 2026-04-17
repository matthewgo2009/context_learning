"""Challenge model (Challenger) wrapper aligned with Self-Evolving ICL framework.

The Challenger πθc generates (Q, E, R) from context C:
  Q = Question, E = Evidence Span extracted from C, R = grading Rubric based on E.

Includes RL-compatible methods (generate_for_rl, generate_group,
compute_per_token_logprobs) mirroring SolverModel, so both models can
be trained with policy gradient.
"""

import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

def _strip_trailing_pad(x: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Trim trailing pad tokens produced by batched `generate`."""
    if pad_id is None or x.numel() == 0:
        return x
    mask = x != pad_id
    nz = mask.nonzero(as_tuple=False)
    if nz.numel() == 0:
        return x[:0]
    last = int(nz[-1].item()) + 1
    return x[:last]


CHALLENGER_SYSTEM_PROMPT = (
    "You are an expert question designer. Given a context passage, your job is to "
    "craft a challenging, non-trivial question that requires deep comprehension of "
    "the context. The context may come from any domain — including but not limited "
    "to science, history, literature, law, medicine, finance, technology, and "
    "everyday reasoning.\n\n"
    "FIRST, in your private scratch-pad, analyze the context and think step-by-step "
    "about what aspects are most important, nuanced, or easily misunderstood.\n\n"
    "THEN, without revealing any of your private thoughts, output exactly the "
    "following three blocks:\n"
    "<question>\n"
    "{A clear, self-contained question that can only be answered correctly by "
    "someone who truly understands the context}\n"
    "</question>\n\n"
    "<evidence>\n"
    "{The exact span extracted from the context that supports the answer}\n"
    "</evidence>\n\n"
    "<rubric>\n"
    "{Evaluation criteria: list the key points a correct answer must cover, "
    "grounded in the evidence span}\n"
    "</rubric>\n\n"
    "Do NOT output anything else — no explanations, no extra markup."
)

CHALLENGER_USER_PROMPT = (
    "Based on the given context, generate one new, challenging question now. "
    "Remember to format the output exactly as instructed: "
    "<question>, <evidence>, then <rubric>."
)


@dataclass
class ChallengerOutput:
    """Parsed (Q, E, R) output from the Challenger model."""
    raw: str
    question: str
    evidence_span: str
    rubric: str


def parse_challenger_output(text: str) -> ChallengerOutput:
    """Parse Challenger raw text into structured (Q, E, R) components."""
    def _extract_tag(tag: str, s: str) -> str:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        m = re.search(pattern, s, re.DOTALL)
        return m.group(1).strip() if m else ""

    return ChallengerOutput(
        raw=text,
        question=_extract_tag("question", text),
        evidence_span=_extract_tag("evidence", text),
        rubric=_extract_tag("rubric", text),
    )


class ChallengeModel:
    """Challenge model that provides context and questions to the Solver."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: Optional[str] = None,
        use_fast: bool = True,
        system_prompt: Optional[str] = None,
        lora_cfg: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            model_name: HuggingFace model identifier.
            device: Device to load model on (None = auto).
            use_fast: Use fast tokenizer.
            system_prompt: Custom system prompt (defaults to CHALLENGER_SYSTEM_PROMPT).
            lora_cfg: Optional PEFT-LoRA config dict; when enabled, the base
                model is wrapped with a LoRA adapter. The reference policy is
                recovered via `with model.disable_adapter():`.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.system_prompt = system_prompt or CHALLENGER_SYSTEM_PROMPT
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=use_fast
        )
        _dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=_dtype,
            trust_remote_code=True,
        ).to(self.device)

        self.lora_cfg = lora_cfg or {}
        self.use_lora = bool(self.lora_cfg.get("enabled", False))
        if self.use_lora:
            self._apply_lora(self.lora_cfg)

    def _apply_lora(self, cfg: Dict[str, Any]) -> None:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            r=int(cfg.get("r", 16)),
            lora_alpha=int(cfg.get("alpha", 32)),
            lora_dropout=float(cfg.get("dropout", 0.05)),
            target_modules=list(cfg.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ])),
            bias=cfg.get("bias", "none"),
            task_type=getattr(TaskType, cfg.get("task_type", "CAUSAL_LM")),
        )
        self.model = get_peft_model(self.model, lora_config)
        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "[Challenger] LoRA enabled (r=%d, alpha=%d): %.2fM / %.2fM params "
            "trainable (%.3f%%)",
            lora_config.r, lora_config.lora_alpha,
            trainable / 1e6, total / 1e6,
            100.0 * trainable / max(total, 1),
        )

    @contextmanager
    def disabled_adapter(self):
        if self.use_lora and hasattr(self.model, "disable_adapter"):
            with self.model.disable_adapter():
                yield
        else:
            yield

    def generate(
        self,
        messages: Optional[List[dict]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        Generate a challenge question.

        If messages is None, uses the default challenger prompt template (B.2).
        Otherwise uses the provided messages directly.

        Args:
            messages: Input in OpenAI chat format. If None, uses default template.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample (vs greedy).
            **kwargs: Additional generation kwargs.

        Returns:
            Generated text containing <question>, <evidence>, and <rubric> blocks.
        """
        if messages is None:
            messages = self.build_challenge_messages()

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        return generated.strip()

    def generate_for_rl(
        self,
        messages: List[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """Generate (Q,E,R) and return token IDs for policy gradient."""
        messages = self._ensure_system_prompt(messages)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        input_len = input_ids.shape[1]
        generated_ids = outputs[:1, input_len:]
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response.strip(), input_ids.detach(), generated_ids.detach()

    def generate_group(
        self,
        messages: List[dict],
        group_size: int = 4,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        """Batched GRPO sampling (num_return_sequences=G)."""
        messages = self._ensure_system_prompt(messages)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=group_size,
                pad_token_id=pad_id,
                **kwargs,
            )

        input_len = input_ids.shape[1]
        results: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        for g in range(group_size):
            gen = _strip_trailing_pad(outputs[g, input_len:], pad_id)
            gen_ids = gen.unsqueeze(0)
            text = self.tokenizer.decode(gen, skip_special_tokens=True)
            results.append((text.strip(), input_ids.detach(), gen_ids.detach()))
        return results

    def compute_per_token_logprobs(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable per-token log-probs for policy gradient."""
        if generated_ids.numel() == 0:
            return torch.zeros(0, device=self.model.device, requires_grad=True)

        full_ids = torch.cat([input_ids, generated_ids], dim=1)
        outputs = self.model(full_ids, use_cache=False)
        input_len = input_ids.shape[1]
        shift_logits = outputs.logits[:, input_len - 1:-1, :]
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(2, generated_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.squeeze(0)

    @torch.no_grad()
    def compute_per_token_logprobs_detached(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Non-differentiable per-token log-probs (for old/reference policy)."""
        if generated_ids.numel() == 0:
            return torch.zeros(0, device=self.model.device)

        full_ids = torch.cat([input_ids, generated_ids], dim=1)
        outputs = self.model(full_ids, use_cache=False)
        input_len = input_ids.shape[1]
        shift_logits = outputs.logits[:, input_len - 1:-1, :]
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(2, generated_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.squeeze(0)

    def compute_per_token_logprobs_batched(
        self,
        input_ids: torch.Tensor,
        generated_ids_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Differentiable batched per-token log-probs (single forward)."""
        G = len(generated_ids_list)
        if G == 0:
            return []
        device = self.model.device
        P = input_ids.shape[1]
        lens = [int(g.shape[1]) for g in generated_ids_list]
        max_L = max(lens) if lens else 0
        if max_L == 0:
            return [
                torch.zeros(0, device=device, requires_grad=True)
                for _ in range(G)
            ]

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        full = torch.full((G, P + max_L), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((G, P + max_L), dtype=torch.long, device=device)
        gen_padded = torch.full((G, max_L), pad_id, dtype=torch.long, device=device)
        for i, gen in enumerate(generated_ids_list):
            L = lens[i]
            full[i, :P] = input_ids[0]
            if L > 0:
                full[i, P:P + L] = gen[0]
                gen_padded[i, :L] = gen[0]
            attn[i, :P + L] = 1

        outputs = self.model(full, attention_mask=attn, use_cache=False)
        shift_logits = outputs.logits[:, P - 1:P + max_L - 1, :]
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        token_lp = log_probs.gather(2, gen_padded.unsqueeze(-1)).squeeze(-1)
        return [token_lp[i, :lens[i]] for i in range(G)]

    @torch.no_grad()
    def compute_per_token_logprobs_detached_batched(
        self,
        input_ids: torch.Tensor,
        generated_ids_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        return self.compute_per_token_logprobs_batched(
            input_ids, generated_ids_list
        )

    @torch.no_grad()
    def compute_per_token_logprobs_ref_batched(
        self,
        input_ids: torch.Tensor,
        generated_ids_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Reference-policy batched per-token log-probs (LoRA disabled)."""
        if not generated_ids_list:
            return []
        with self.disabled_adapter():
            return self.compute_per_token_logprobs_batched(
                input_ids, generated_ids_list
            )

    @torch.no_grad()
    def compute_per_token_logprobs_ref(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Reference-policy per-token log-probs via disabled LoRA adapter."""
        if generated_ids.numel() == 0:
            return torch.zeros(0, device=self.model.device)
        with self.disabled_adapter():
            return self.compute_per_token_logprobs_detached(input_ids, generated_ids)

    def save_adapter(self, path) -> None:
        """Save LoRA adapter weights (or the full model if LoRA is disabled)."""
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

    def load_adapter(self, path) -> None:
        if not self.use_lora:
            raise RuntimeError("load_adapter called but LoRA is not enabled")
        from peft import PeftModel
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") \
            else self.model
        self.model = PeftModel.from_pretrained(base, str(path)).to(self.device)

    def build_challenge_messages(self) -> List[dict]:
        """Build the default challenger prompt messages following template B.2."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": CHALLENGER_USER_PROMPT},
        ]

    def build_context_messages(self, context: str) -> List[dict]:
        """Build challenger messages with a specific context C."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\n{CHALLENGER_USER_PROMPT}"},
        ]

    def format_prompt(self, messages: List[dict]) -> List[dict]:
        """Injects the challenger system prompt if no system message is present."""
        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            return messages
        return [{"role": "system", "content": self.system_prompt}] + messages

    def _ensure_system_prompt(self, messages: List[dict]) -> List[dict]:
        """Inject challenger system prompt if missing."""
        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            return messages
        return [{"role": "system", "content": self.system_prompt}] + messages
