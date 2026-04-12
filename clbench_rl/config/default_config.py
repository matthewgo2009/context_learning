"""Default configuration for CL-bench RL adversarial training.

Aligned with: "Self-Evolving In-Context Learning via Asymmetric Adversarial Play"
Default models: Qwen3-4B-Instruct-2507 for both Challenger and Solver.
"""

from typing import Any, Dict, Optional


def get_default_config() -> Dict[str, Any]:
    """Return default configuration dict."""
    return {
        "data": {
            "split": "train",
            "max_samples": None,
            "subset": None,
            "cache_dir": None,
        },
        "challenge_model": {
            "model_name": "Qwen/Qwen3-4B-Instruct-2507",
            "device": None,
            "max_new_tokens": 1024,
            "temperature": 0.7,
        },
        "solver_model": {
            "model_name": "Qwen/Qwen3-4B-Instruct-2507",
            "device": None,
            "max_new_tokens": 2048,
            "temperature": 0.7,
        },
        "reward": {
            "use_llm_judge": True,
            "judge_model": "gpt-4o",
            "judge_temperature": 0.1,
            # Challenger reward weights w1..w5 (paper: dynamic hyperparameters)
            "w1_adversarial": 1.0,
            "w2_repetition": 0.3,
            "w3_format": 0.2,
            "w4_relevance": 0.3,
            "w5_rubric": 0.2,
            # BLEU-clustering repetition penalty (Appendix B.4)
            "bleu_distance_threshold": 0.5,
            "repetition_batch_size": 16,
            # Dynamic weight scheduling (paper: "dynamic hyperparameters")
            "use_dynamic_weights": True,
            "w1_init": 0.3,
            "w1_final": 1.0,
            "w3_init": 0.5,
            "w3_final": 0.1,
        },
        "training": {
            "train_solver": True,
            "train_challenge": True,
            "epochs": 3,
            "lr": 1e-5,
            "solver_lr": 1e-5,
            "challenger_lr": 5e-6,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "checkpoint_dir": "checkpoints",
            "save_every": 500,
            "log_every": 10,
            "ref_sync_every": 200,
        },
        "grpo": {
            "group_size": 4,
            "clip_eps": 0.2,
            "kl_beta": 0.04,
            "adv_eps": 1e-8,
            "max_grad_norm": 1.0,
            "mu_iterations": 1,
        },
    }


def merge_config(user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge user config over defaults with shallow merge per top-level key."""
    base = get_default_config()
    if not user_config:
        return base
    for key, val in user_config.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            base[key] = {**base[key], **val}
        else:
            base[key] = val
    return base
