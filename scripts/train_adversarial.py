#!/usr/bin/env python3
"""Adversarial self-play training: Solver + Challenger dual GRPO.

Usage (single-node 8 GPU via torchrun):
    torchrun --nproc_per_node=8 scripts/train_adversarial.py

Usage (with DeepSpeed via accelerate):
    accelerate launch --config_file configs/accelerate_config.yaml \
        scripts/train_adversarial.py

All training hyperparameters come from default_config.py and can be
overridden via CLI args below.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Adversarial self-play training")

    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                    help="Base model for both Solver and Challenger")
    p.add_argument("--solver-model", type=str, default=None,
                    help="Override model for Solver (defaults to --model)")
    p.add_argument("--challenger-model", type=str, default=None,
                    help="Override model for Challenger (defaults to --model)")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5, help="Solver LR")
    p.add_argument("--challenger-lr", type=float, default=5e-6, help="Challenger LR")
    p.add_argument("--group-size", type=int, default=4, help="GRPO group size G")
    p.add_argument("--kl-beta", type=float, default=0.04, help="KL penalty coefficient")
    p.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")

    p.add_argument("--max-samples", type=int, default=None,
                    help="Limit dataset size (None = full)")
    p.add_argument("--data-split", type=str, default="train")
    p.add_argument("--data-cache-dir", type=str, default=None)

    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ref-sync-every", type=int, default=200)

    p.add_argument("--use-llm-judge", action="store_true", default=True,
                    help="Use frozen LLM as Judge (default: True, requires OPENAI_API_KEY)")
    p.add_argument("--no-llm-judge", dest="use_llm_judge", action="store_false",
                    help="Disable LLM Judge, use heuristic fallback")
    p.add_argument("--judge-model", type=str, default="gpt-4o",
                    help="Judge model name (default: gpt-4o)")
    p.add_argument("--openai-api-key", type=str, default=None,
                    help="OpenAI API key (or set OPENAI_API_KEY env var)")

    p.add_argument("--output-dir", type=str, default="outputs")

    return p.parse_args()


def main():
    args = parse_args()

    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    solver_model = args.solver_model or args.model
    challenger_model = args.challenger_model or args.model

    config = {
        "data": {
            "split": args.data_split,
            "max_samples": args.max_samples,
            "cache_dir": args.data_cache_dir,
        },
        "challenge_model": {
            "model_name": challenger_model,
        },
        "solver_model": {
            "model_name": solver_model,
        },
        "reward": {
            "use_llm_judge": args.use_llm_judge,
            "judge_model": args.judge_model,
        },
        "training": {
            "epochs": args.epochs,
            "solver_lr": args.lr,
            "challenger_lr": args.challenger_lr,
            "checkpoint_dir": args.checkpoint_dir,
            "save_every": args.save_every,
            "log_every": args.log_every,
            "ref_sync_every": args.ref_sync_every,
        },
        "grpo": {
            "group_size": args.group_size,
            "kl_beta": args.kl_beta,
            "clip_eps": args.clip_eps,
        },
    }

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        logger.info("Config: %s", json.dumps(config, indent=2, default=str))

    from clbench_rl.trainer.adversarial_trainer import AdversarialTrainer

    trainer = AdversarialTrainer(config=config)
    metrics = trainer.train()

    if local_rank == 0:
        logger.info("Training complete.")
        print("\n" + "=" * 60)
        print("Adversarial Self-Play Training Complete")
        print("=" * 60)
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                print(f"  {k:30s}: {v:.6f}")
            else:
                print(f"  {k:30s}: {v}")
        print("=" * 60)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "final_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
