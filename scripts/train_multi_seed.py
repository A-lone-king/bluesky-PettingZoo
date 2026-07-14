"""Multi-seed training script for paper-level reproducibility.

Run PPO/SAC/TD3/DDPG across multiple random seeds and aggregate results.

Usage:
    python scripts/train_multi_seed.py --scenario HorizontalCR --algorithm PPO --timesteps 500000
    python scripts/train_multi_seed.py --scenario HorizontalCR --algorithm PPO --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from bluesky_pettingzoo.training.multi_seed import MultiSeedTrainer, SeedResult
from scripts.train_ppo_scenarios import (
    _resolve_scenario,
    make_scenario_env_factory,
    _get_algo_class,
    _make_model,
    _resolve_device,
    _safe_get,
)

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from tests.helpers.env_factory import make_config, write_rewards_yaml


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
    "PlanWaypoint": "PlanWaypointScenario",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Multi-seed training for BlueSky scenarios")
    parser.add_argument("--scenario", type=str, default="HorizontalCR",
                        choices=list(SCENARIO_MAP.keys()),
                        help="Scenario to train on")
    parser.add_argument("--algorithm", type=str, default="PPO",
                        choices=["PPO", "SAC", "TD3", "DDPG"],
                        help="RL algorithm to use")
    parser.add_argument("--action-space", type=str, default="discrete",
                        choices=["discrete", "continuous"],
                        help="Action space type")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps per seed")
    parser.add_argument("--num-aircraft", type=int, default=3,
                        help="Number of aircraft in scenario")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max steps per episode")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="List of random seeds to use")
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Base directory for model checkpoints")
    parser.add_argument("--results-dir", type=str, default="results/multi_seed",
                        help="Directory for saving summary results")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Number of evaluation episodes per seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level")
    return parser.parse_args(argv)


def _create_train_fn(args: argparse.Namespace):
    """Create a training function that trains a single seed."""

    def train_fn(seed: int, seed_save_dir: Path) -> SeedResult:
        """Train on a single seed and return evaluation results."""
        algo_cls = _get_algo_class(args.algorithm)
        action_space = args.action_space
        if args.algorithm in ("SAC", "TD3", "DDPG"):
            action_space = "continuous"

        scenario = _resolve_scenario(args.scenario, args.num_aircraft, seed)
        if action_space == "continuous":
            scenario.action_space_type = "continuous"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            factory = make_scenario_env_factory(
                tmp_path, scenario, args.num_aircraft, args.max_steps,
            )
            env = factory()

            mock_args = argparse.Namespace(
                algorithm=args.algorithm,
                seed=seed,
                lr=_safe_get(args, "lr", 3e-4),
                n_steps=_safe_get(args, "n_steps", 2048),
                batch_size=_safe_get(args, "batch_size", 256),
                n_epochs=_safe_get(args, "n_epochs", 4),
                gamma=_safe_get(args, "gamma", 0.99),
                gae_lambda=_safe_get(args, "gae_lambda", 0.95),
                device=args.device,
                verbose=args.verbose,
                tensorboard_log_dir=None,
                num_envs=1,
            )

            model = _make_model(algo_cls, env, mock_args)

            print(f"\n  Training seed {seed}...")
            model.learn(total_timesteps=args.timesteps)

            model_path = seed_save_dir / "final_model.zip"
            model.save(str(model_path))

            rewards, steps_list, arrivals, nmacs = _evaluate_model(
                model, env, args.eval_episodes, args.max_steps, seed
            )

            env.close()

            return SeedResult(
                seed=seed,
                mean_reward=float(np.mean(rewards)) if rewards else 0.0,
                std_reward=float(np.std(rewards)) if rewards else 0.0,
                mean_arrival_rate=arrivals / len(rewards) if rewards else 0.0,
                mean_nmac_rate=nmacs / len(rewards) if rewards else 0.0,
                mean_steps=float(np.mean(steps_list)) if steps_list else 0.0,
                model_path=str(model_path),
                eval_episodes=args.eval_episodes,
            )

    return train_fn


def _evaluate_model(
    model,
    env,
    num_episodes: int,
    max_steps: int,
    seed: int,
) -> tuple[list[float], list[int], int, int]:
    """Evaluate model and return aggregated metrics."""
    rng = np.random.RandomState(seed)
    rewards: list[float] = []
    steps_list: list[int] = []
    arrivals = 0
    nmacs = 0

    for _ in range(num_episodes):
        obs, _ = env.reset(seed=rng.randint(0, 2**31))
        total_reward = 0.0
        episode_steps = 0
        arrived = False

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            episode_steps = step + 1

            if terminated or truncated:
                arrived = total_reward > 0
                break

        rewards.append(total_reward)
        steps_list.append(episode_steps)
        if arrived:
            arrivals += 1
        else:
            nmacs += 1

    return rewards, steps_list, arrivals, nmacs


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    print(f"\nMulti-seed Training: {args.scenario} × {args.algorithm}")
    print(f"Timesteps per seed: {args.timesteps:,}")
    print(f"Seeds: {args.seeds or MultiSeedTrainer.DEFAULT_SEEDS}")
    print(f"Results will be saved to: {args.results_dir}")

    train_fn = _create_train_fn(args)
    trainer = MultiSeedTrainer(
        scenario=args.scenario,
        algorithm=args.algorithm,
        train_fn=train_fn,
        seeds=args.seeds,
        save_dir=args.save_dir,
        results_dir=args.results_dir,
    )

    summary = trainer.train_all()
    print(f"\n{trainer.format_summary(summary)}")


if __name__ == "__main__":
    main()