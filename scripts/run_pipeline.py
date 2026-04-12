#!/usr/bin/env python3
"""Run CL-bench RL pipeline: load data, run env, report rewards."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_dry_run(max_samples: int) -> None:
    """Dry run: test data loading and reward logic without loading models."""
    from clbench_rl.data.loader import CLBenchDataLoader
    from clbench_rl.rewards.rubrics_reward import RubricsReward

    def _context_from_messages(messages: list) -> str:
        parts = []
        for m in messages:
            if isinstance(m, dict) and m.get("content"):
                parts.append(m["content"])
        return "\n".join(parts)

    print("Dry run: loading dataset...")
    loader = CLBenchDataLoader(split="train", max_samples=max_samples)
    loader.load()
    reward_fn = RubricsReward()

    total_r = 0.0
    for i, sample in enumerate(loader):
        ctx = _context_from_messages(sample.get("messages", []))
        r = reward_fn.compute_solver_reward(
            answer="This is a placeholder answer for dry run.",
            rubrics=sample["rubrics"],
            context=ctx,
            metadata=sample["metadata"],
        )
        total_r += r.total
    avg = total_r / max_samples if max_samples else 0.0
    print(f"Dry run completed. Processed {max_samples} samples.")
    print(f"Mean heuristic solver reward: {avg:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run CL-bench RL pipeline")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Max samples to process (for quick validation)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name for both Challenge and Solver (local small model)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test data loading and rewards without loading models",
    )
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run(args.max_samples)
        return

    from clbench_rl.trainer.reinforce_trainer import ReinforceTrainer

    config = {
        "data": {
            "max_samples": args.max_samples,
        },
        "challenge_model": {"model_name": args.model},
        "solver_model": {"model_name": args.model},
        "training": {
            "checkpoint_dir": args.checkpoint_dir,
            "save_every": None,
            "log_every": 2,
        },
    }

    trainer = ReinforceTrainer(config=config)
    metrics = trainer.train()

    print("\n" + "=" * 50)
    print("Pipeline completed.")
    print(f"Mean J_score(A,R):     {metrics['mean_j_score']:.4f}")
    print(f"Mean Solver Reward:    {metrics['mean_solver_reward']:.4f}")
    print(f"Mean Challenge Reward: {metrics['mean_challenge_reward']:.4f}")
    print(f"Mean R_adv:            {metrics['mean_r_adv']:.4f}")
    print(f"Mean R_rep:            {metrics['mean_r_rep']:.4f}")
    print(f"Mean R_fmt:            {metrics['mean_r_fmt']:.4f}")
    print(f"Mean R_rel:            {metrics['mean_r_rel']:.4f}")
    print(f"Mean R_rubric:         {metrics['mean_r_rubric']:.4f}")
    print(f"Episodes:              {metrics['num_episodes']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
