#!/usr/bin/env python
"""Root training entry point for bluesky-pettingzoo.

Delegates to scripts/train_ppo_scenarios.py with YAML config support.

Usage:
    python train.py --config configs/quick.yaml
    python train.py --scenario HorizontalCR --timesteps 100000
    python train.py --scenario HorizontalCR --algorithm SAC --timesteps 200000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments with optional YAML config override."""
    parser = argparse.ArgumentParser(
        description="Train RL agent on BlueSky ATM scenario",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (overrides all other args)",
    )
    parser.add_argument("--scenario", type=str, default="HorizontalCR")
    parser.add_argument("--algorithm", type=str, default="PPO",
                        choices=["PPO", "SAC", "TD3", "DDPG"])
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--num-aircraft", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--norm-reward", action="store_true", default=False)
    parser.add_argument("--tensorboard", action="store_true", default=False)
    return parser.parse_args()


def build_args_from_config(config_path: str) -> argparse.Namespace:
    """Load a YAML config and convert to an argparse.Namespace."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    scenario_cfg = cfg.get("scenario", {})
    training_cfg = cfg.get("training", {})
    checkpoint_cfg = cfg.get("checkpoint", {})
    logging_cfg = cfg.get("logging", {})

    return argparse.Namespace(
        scenario=scenario_cfg.get("name", "HorizontalCR"),
        algorithm=training_cfg.get("algorithm", "PPO"),
        timesteps=training_cfg.get("timesteps", 50_000),
        num_aircraft=scenario_cfg.get("num_aircraft", 3),
        max_steps=scenario_cfg.get("max_steps", 50),
        seed=training_cfg.get("seed", 42),
        save_dir=checkpoint_cfg.get("save_dir", "models"),
        device="cpu",
        verbose=0,
        norm_reward=training_cfg.get("norm_reward", False),
        tensorboard=logging_cfg.get("tensorboard", False),
        resume=None,
        action_space="discrete",
        lr=training_cfg.get("learning_rate", 3e-4),
        n_steps=training_cfg.get("n_steps", 2048),
        batch_size=training_cfg.get("batch_size", 256),
        n_epochs=training_cfg.get("n_epochs", 4),
        gamma=training_cfg.get("gamma", 0.99),
        gae_lambda=training_cfg.get("gae_lambda", 0.95),
        num_envs=training_cfg.get("num_envs", 1),
        render=False,
        tensorboard_log_dir=logging_cfg.get("tensorboard_log_dir", "runs"),
    )


def main() -> int:
    """Entry point: load config if provided, then delegate to training script."""
    args = parse_args()

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: config file not found: {config_path}", file=sys.stderr)
            return 1
        args = build_args_from_config(str(config_path))
        print(f"Loaded config from {config_path}")

    # Inject tensorboard_log_dir into args if --tensorboard is set
    if not hasattr(args, "tensorboard_log_dir"):
        args.tensorboard_log_dir = "runs"

    # Delegate to the training script
    from scripts.train_ppo_scenarios import train_scenario

    print(f"Training {args.algorithm} on {args.scenario}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Aircraft: {args.num_aircraft}, Max steps: {args.max_steps}")
    print(f"  Seed: {args.seed}")

    result = train_scenario(args)

    print(f"\nTraining complete: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
