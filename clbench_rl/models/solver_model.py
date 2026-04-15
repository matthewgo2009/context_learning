"""Solver model (Student) wrapper for generating answers from context."""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    ):
        """
        Args:
            model_name: HuggingFace model identifier.
            device: Device to load model on (None = auto).
            use_fast: Use fast tokenizer.
            system_prompt: Custom system prompt (defaults to SOLVER_SYSTEM_PROMPT).
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.system_prompt = system_prompt or SOLVER_SYSTEM_PROMPT
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=use_fast
        )
        if self.device.startswith("cuda"):
            _dmap = {"": self.device} if ":" in self.device else "auto"
            _dtype = torch.bfloat16
        else:
            _dmap = None
            _dtype = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=_dtype,
            device_map=_dmap,
            trust_remote_code=True,
        )
        if not self.device.startswith("cuda"):
            self.model = self.model.to(self.device)

    def enable_gradient_checkpointing(self):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

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
        Generate a group of G responses for the same prompt (GRPO sampling).

        Each response is sampled independently. Returns G tuples of
        (response_text, input_ids, generated_ids) for subsequent log-prob
        computation and policy gradient.

        Args:
            messages: OpenAI chat format messages.
            group_size: Number of responses G to sample per prompt.
            max_new_tokens: Max tokens to generate per response.
            temperature: Sampling temperature.

        Returns:
            List of G tuples: [(text, input_ids, generated_ids), ...]
        """
        messages = self._ensure_system_prompt(messages)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]

        results = []
        with torch.no_grad():
            for _ in range(group_size):
                outputs = self.model.generate(
                    **{k: v.clone() for k, v in inputs.items()},
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )
                input_len = input_ids.shape[1]
                gen_ids = outputs[:1, input_len:]
                text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                results.append((text.strip(), input_ids.detach(), gen_ids.detach()))

        return results

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

    def _ensure_system_prompt(self, messages: List[dict]) -> List[dict]:
        """Inject solver system prompt if no system message is present."""
        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            return messages
        return [{"role": "system", "content": self.system_prompt}] + messages
