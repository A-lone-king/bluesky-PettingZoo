#!/usr/bin/env python
"""Root evaluation entry point for bluesky-pettingzoo.

Delegates to scripts/evaluate_baselines.py for baseline comparison.

Usage:
    python eval.py --scenario HorizontalCR --episodes 20
    python eval.py --scenario HorizontalCR --model models/HorizontalCR/PPO/checkpoint_final.zip
"""
from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained models and baselines",
    )
    parser.add_argument("--scenario", type=str, default="HorizontalCR")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--num-aircraft", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    """Entry point: delegate to the evaluation script."""
    args = parse_args()

    # Delegate to the evaluation script
    from scripts.evaluate_baselines import run_evaluation

    print(f"Evaluating on {args.scenario} ({args.episodes} episodes)")
    if args.model:
        print(f"  Model: {args.model}")

    results = run_evaluation(args)

    print(f"\nEvaluation complete: {len(results)} strategies evaluated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
