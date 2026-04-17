"""Solver model (Student) wrapper for generating answers from context."""

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

def _strip_trailing_pad(x: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Trim the trailing pad tokens produced by batched `generate`."""
    if pad_id is None or x.numel() == 0:
        return x
    mask = x != pad_id
    nz = mask.nonzero(as_tuple=False)
    if nz.numel() == 0:
        return x[:0]
    last = int(nz[-1].item()) + 1
    return x[:last]


SOLVER_SYSTEM_PROMPT = (
    "You are given a context and a question. Read the context carefully, "
    "reason step by step, and provide a clear, well-supported answer. "
    "Base your answer strictly on the information in the context. "
    "Wrap your final answer in <answer>...</answer> tags."
)


class SolverModel:
    """Solver model that answers questions based on provided context."""

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
            system_prompt: Custom system prompt (defaults to SOLVER_SYSTEM_PROMPT).
            lora_cfg: Optional PEFT-LoRA config dict; when enabled, the base
                model is wrapped with a LoRA adapter. The reference policy is
                recovered via `with model.disable_adapter():` — no separate
                frozen copy is needed.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.system_prompt = system_prompt or SOLVER_SYSTEM_PROMPT
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
        """Wrap `self.model` with a PEFT LoRA adapter.

        Only LoRA params will have `requires_grad=True`, so the optimizer
        naturally sees a ~50MB slice instead of the full 4B model.
        """
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
        trainable, total = self._count_trainable()
        logger.info(
            "[Solver] LoRA enabled (r=%d, alpha=%d): %.2fM / %.2fM params "
            "trainable (%.3f%%)",
            lora_config.r, lora_config.lora_alpha,
            trainable / 1e6, total / 1e6,
            100.0 * trainable / max(total, 1),
        )

    def _count_trainable(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    @contextmanager
    def disabled_adapter(self):
        """Temporarily disable the LoRA adapter (→ base/reference policy).

        Falls back to a no-op if LoRA is not enabled.
        """
        if self.use_lora and hasattr(self.model, "disable_adapter"):
            with self.model.disable_adapter():
                yield
        else:
            yield

    def build_solver_messages(
        self, question: str, context: str = ""
    ) -> List[dict]:
        """
        Build solver messages with system prompt, context, and question.

        Args:
            question: The question to answer.
            context: Optional context passage to ground the answer in.

        Returns:
            Messages in OpenAI chat format with system + user roles.
        """
        user_content = question
        if context:
            user_content = f"Context:\n{context}\n\nQuestion:\n{question}"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def generate(
        self,
        messages: List[dict],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        Generate answer (inference only, no gradient).

        Automatically injects solver system prompt if no system message is present.

        Args:
            messages: OpenAI chat format [{"role": "system", "content": "..."}, ...].
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample.

        Returns:
            Generated response text.
        """
        messages = self._ensure_system_prompt(messages)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
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
        response = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        return response.strip()

    def generate_for_rl(
        self,
        messages: List[dict],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Generate answer and return token IDs for subsequent differentiable
        log-prob computation. Generation itself is non-differentiable.

        Args:
            messages: OpenAI chat format messages.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample.

        Returns:
            (response_text, input_ids [1, prompt_len], generated_ids [1, gen_len])
        """
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
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        """
        Batched GRPO sampling: single `generate` call with
        `num_return_sequences=G`. Replaces a Python for-loop of G sequential
        generates. Trailing pad tokens are stripped per trajectory so that
        `compute_per_token_logprobs` does not charge loss on padding.
        """
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

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> List[str]:
        """Batched text generation for a list of ready-to-use prompt strings.

        Uses **left-padding** (required so that the newly generated tokens
        are appended at a consistent position for every sequence). Restores
        the tokenizer's original `padding_side` on exit.
        """
        if not prompts:
            return []
        tok = self.tokenizer
        old_side = tok.padding_side
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        try:
            inputs = tok(
                prompts, return_tensors="pt", padding=True,
            ).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=tok.eos_token_id,
                    **kwargs,
                )
            input_len = inputs["input_ids"].shape[1]
            texts: List[str] = []
            for i in range(len(prompts)):
                gen = outputs[i, input_len:]
                texts.append(tok.decode(gen, skip_special_tokens=True).strip())
            return texts
        finally:
            tok.padding_side = old_side

    def compute_sequence_logprob(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable forward pass to compute mean log-probability of the
        generated sequence, suitable for policy gradient.

        Args:
            input_ids: Prompt token IDs, shape [1, prompt_len].
            generated_ids: Generated token IDs, shape [1, gen_len].

        Returns:
            Scalar tensor: mean log P(generated | prompt), with gradient.
        """
        if generated_ids.numel() == 0:
            return torch.tensor(0.0, device=self.model.device, requires_grad=True)

        full_ids = torch.cat([input_ids, generated_ids], dim=1)
        outputs = self.model(full_ids, use_cache=False)
        input_len = input_ids.shape[1]
        shift_logits = outputs.logits[:, input_len - 1:-1, :]
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(2, generated_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.mean(dim=-1).squeeze(0)

    def compute_per_token_logprobs(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable forward pass returning per-token log-probabilities.

        Args:
            input_ids: Prompt token IDs, shape [1, prompt_len].
            generated_ids: Generated token IDs, shape [1, gen_len].

        Returns:
            Tensor of shape [gen_len] with per-token log P, with gradient.
        """
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
        """
        Non-differentiable per-token log-probs (for old policy / reference model).

        Args:
            input_ids: Prompt token IDs, shape [1, prompt_len].
            generated_ids: Generated token IDs, shape [1, gen_len].

        Returns:
            Tensor of shape [gen_len] with per-token log P, detached.
        """
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
        """Differentiable per-token log-probs for G trajectories sharing the
        same prompt, in a **single forward pass**.

        Right-pads the generated part and sets `attention_mask` so padded
        positions do not leak into attention. Returns one [L_i] tensor per
        trajectory, all sharing a single computation graph — aggregate the
        losses and call `.backward()` once.
        """
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
        """No-grad batched per-token log-probs."""
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
        """
        Reference-policy per-token log-probs.

        With LoRA: temporarily disables the adapter so the forward pass runs
        through the frozen base model. Without LoRA: identical to
        `compute_per_token_logprobs_detached` (caller is expected to hold a
        separate frozen ref model in that case).
        """
        if generated_ids.numel() == 0:
            return torch.zeros(0, device=self.model.device)
        with self.disabled_adapter():
            return self.compute_per_token_logprobs_detached(input_ids, generated_ids)

    def save_adapter(self, path) -> None:
        """Save LoRA adapter weights (or the full model if LoRA is disabled)."""
        if self.use_lora:
            self.model.save_pretrained(str(path))
        else:
            self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

    def load_adapter(self, path) -> None:
        """Load a previously-saved LoRA adapter into the current base model."""
        if not self.use_lora:
            raise RuntimeError("load_adapter called but LoRA is not enabled")
        from peft import PeftModel
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") \
            else self.model
        self.model = PeftModel.from_pretrained(base, str(path)).to(self.device)

    def _ensure_system_prompt(self, messages: List[dict]) -> List[dict]:
        """Inject solver system prompt if no system message is present."""
        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            return messages
        return [{"role": "system", "content": self.system_prompt}] + messages
