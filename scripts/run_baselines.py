"""Baseline comparison — train PPO and compare against Random and RuleBased.

Usage:
    python scripts/run_baselines.py --scenario HorizontalCR --timesteps 50000 --episodes 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


SCENARIO_MAP: dict[str, str] = {
    "HorizontalCR": "HorizontalCRScenario",
    "VerticalCR": "VerticalCRScenario",
    "SectorCR": "SectorCRScenario",
    "WaypointNav": "WaypointNavScenario",
    "Merge": "MergeScenario",
    "Descent": "DescentScenario",
    "StaticObstacle": "StaticObstacleScenario",
    "SectorCapacity": "SectorCapacityScenario",
    "RouteNav": "RouteNavScenario",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train PPO and compare against baselines")
    parser.add_argument("--scenario", type=str, default="HorizontalCR",
                        choices=list(SCENARIO_MAP.keys()),
                        help="Scenario to run on")
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="PPO training timesteps")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Evaluation episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Directory to save models")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max steps per episode")
    parser.add_argument("--num-aircraft", type=int, default=3,
                        help="Number of aircraft")
    return parser.parse_args(argv)


def run_baselines(args: argparse.Namespace) -> None:
    """Train PPO, evaluate all strategies, print comparison table."""
    from scripts.evaluate_baselines import run_evaluation as run_eval
    from scripts.train_ppo_scenarios import train_scenario

    # Step 1: Train PPO
    print(f"Training PPO on {args.scenario} for {args.timesteps} timesteps...")
    train_args = argparse.Namespace(
        scenario=args.scenario,
        timesteps=args.timesteps,
        seed=args.seed,
        save_dir=args.save_dir,
        resume=None,
        max_steps=args.max_steps,
        num_aircraft=args.num_aircraft,
        algorithm="PPO",
    )
    train_scenario(train_args)

    # Step 2: Evaluate all strategies
    model_path = str(Path(args.save_dir) / args.scenario / "PPO" / "checkpoint_final.zip")
    eval_args = argparse.Namespace(
        scenario=args.scenario,
        model=model_path,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        num_aircraft=args.num_aircraft,
    )
    run_eval(eval_args)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_baselines(args)


if __name__ == "__main__":
    main()
