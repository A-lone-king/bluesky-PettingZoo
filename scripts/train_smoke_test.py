"""Smoke test: verify PPO can learn in BlueSkyMARLEnv.

Uses BlueSkyWrapper + WaypointNavScenario (1 aircraft, no conflicts).
The agent must learn to turn toward its waypoint to collect arrival rewards.

Success criteria:
  - Training completes without error over 10k timesteps
  - Mean episode reward improves from first to last evaluation
  - Final mean reward > initial mean reward (learning signal present)

Usage:
    python scripts/train_smoke_test.py
    python scripts/train_smoke_test.py --multi-scenario
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path
from typing import Any

# Ensure project root is on the path for test helpers
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from bluesky_pettingzoo.training.progress import ProgressCallback
from bluesky_pettingzoo.wrappers.single_agent import SingleAgentGymWrapper
from tests.helpers.env_factory import make_config, write_rewards_yaml

# Core scenarios for multi-scenario validation
CORE_SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "HorizontalCR",
        "scenario_cls": "HorizontalCRScenario",
        "num_aircraft": 3,
        "max_steps": 30,
    },
    {
        "name": "SectorCR",
        "scenario_cls": "SectorCRScenario",
        "num_aircraft": 3,
        "max_steps": 30,
    },
    {
        "name": "WaypointNav",
        "scenario_cls": "WaypointNavScenario",
        "num_aircraft": 1,
        "max_steps": 50,
    },
]


def _make_env(tmp_path: Path) -> SingleAgentGymWrapper:
    """Create a SingleAgentGymWrapper with WaypointNav + FakeBlueSky."""
    config = make_config(initial_count=1, max_steps=50)
    rewards_path = write_rewards_yaml(tmp_path)
    config["_rewards_yaml"] = str(rewards_path)

    with open(rewards_path, encoding="utf-8") as f:
        rewards_cfg = yaml.safe_load(f)
    merged = {**config, **rewards_cfg}

    wrapper = BlueSkyWrapper(config)
    obs_manager = ObservationManager(config)
    action_translator = ActionTranslator(config)
    calc = RewardCalculator()
    calc.register(ConflictPenalty(merged), weight=1.0)
    calc.register(SmoothnessPenalty(merged), weight=0.5)
    calc.register(EfficiencyReward(merged), weight=0.3)

    env = BlueSkyMARLEnv(
        config=config,
        wrapper=wrapper,
        observation_manager=obs_manager,
        action_translator=action_translator,
        reward_calculator=calc,
        rewards_config=rewards_cfg,
        scenario=WaypointNavScenario(num_aircraft=1, seed=42),
    )
    return SingleAgentGymWrapper(env, ego_agent="AC000")


def _evaluate(model: Any, env: Any, n_episodes: int = 5) -> float:
    """Evaluate model over n_episodes and return mean reward."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        for _ in range(60):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    return float(np.mean(rewards))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="PPO smoke test for BlueSkyMARLEnv")
    parser.add_argument(
        "--multi-scenario", action="store_true", default=False,
        help="Train on multiple core scenarios (HorizontalCR, SectorCR, WaypointNav)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=10_000,
        help="Total training timesteps (default: 10000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for CSV logs (default: temp dir)",
    )
    return parser.parse_args(argv)


def _resolve_scenario(name: str, num_aircraft: int, seed: int) -> Any:
    """Import and instantiate a scenario class by name."""
    from bluesky_pettingzoo.envs.scenarios import horizontal_cr, sector_cr, waypoint_nav

    module_map: dict[str, Any] = {
        "HorizontalCRScenario": horizontal_cr,
        "SectorCRScenario": sector_cr,
        "WaypointNavScenario": waypoint_nav,
    }
    scenario_map: dict[str, str] = {
        "HorizontalCR": "HorizontalCRScenario",
        "SectorCR": "SectorCRScenario",
        "WaypointNav": "WaypointNavScenario",
    }
    cls_name = scenario_map[name]
    mod = module_map[cls_name]
    cls = getattr(mod, cls_name)
    return cls(num_aircraft=num_aircraft, seed=seed)


def _make_env_for_scenario(
    tmp_path: Path,
    scenario_name: str,
    num_aircraft: int,
    max_steps: int,
    seed: int = 42,
) -> SingleAgentGymWrapper:
    """Create a SingleAgentGymWrapper for a given scenario."""
    scenario = _resolve_scenario(scenario_name, num_aircraft, seed)
    config = make_config(initial_count=num_aircraft, max_steps=max_steps)
    rewards_path = write_rewards_yaml(tmp_path)
    config["_rewards_yaml"] = str(rewards_path)

    with open(rewards_path, encoding="utf-8") as f:
        rewards_cfg = yaml.safe_load(f)
    merged = {**config, **rewards_cfg}

    wrapper = BlueSkyWrapper(config)
    obs_manager = ObservationManager(config)
    action_translator = ActionTranslator(config)
    calc = RewardCalculator()
    calc.register(ConflictPenalty(merged), weight=1.0)
    calc.register(SmoothnessPenalty(merged), weight=0.5)
    calc.register(EfficiencyReward(merged), weight=0.3)

    env = BlueSkyMARLEnv(
        config=config,
        wrapper=wrapper,
        observation_manager=obs_manager,
        action_translator=action_translator,
        reward_calculator=calc,
        rewards_config=rewards_cfg,
        scenario=scenario,
    )
    return SingleAgentGymWrapper(env, ego_agent="AC000")


def save_training_curve_csv(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save training curve results to CSV.

    Args:
        results: List of dicts with keys: scenario, initial_reward,
            final_reward, improvement, timesteps
        output_path: Path to save the CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scenario", "initial_reward", "final_reward", "improvement", "timesteps"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def train_single_scenario(
    scenario_cfg: dict[str, Any],
    timesteps: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Train PPO on a single scenario and return results.

    Args:
        scenario_cfg: Scenario configuration dict with name, num_aircraft, max_steps
        timesteps: Total training timesteps
        seed: Random seed
        output_dir: Output directory for model and logs

    Returns:
        Dict with scenario, initial_reward, final_reward, improvement, timesteps
    """
    from stable_baselines3 import PPO

    scenario_name = scenario_cfg["name"]
    num_aircraft = scenario_cfg["num_aircraft"]
    max_steps = scenario_cfg["max_steps"]

    print(f"\n{'='*60}")
    print(f"Training PPO on {scenario_name}")
    print(f"  Aircraft: {num_aircraft}, Max steps: {max_steps}")
    print(f"  Timesteps: {timesteps:,}, Seed: {seed}")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        env = _make_env_for_scenario(tmp_path, scenario_name, num_aircraft, max_steps, seed)

        # Create PPO model
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            learning_rate=3e-4,
            verbose=0,
            device="cpu",
            seed=seed,
        )

        # Evaluate before training
        initial_reward = _evaluate(model, env, n_episodes=5)
        print(f"  [Before] Mean reward: {initial_reward:.2f}")

        # Train
        model.learn(total_timesteps=timesteps, callback=[ProgressCallback()])

        # Evaluate after training
        final_reward = _evaluate(model, env, n_episodes=5)
        print(f"  [After]  Mean reward: {final_reward:.2f}")

        # Save model
        model_dir = output_dir / scenario_name / "PPO"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "checkpoint_final.zip"
        model.save(str(model_path))
        print(f"  Model saved to {model_path}")

        env.close()

    improvement = final_reward - initial_reward
    print(f"  Improvement: {improvement:+.2f}")

    return {
        "scenario": scenario_name,
        "initial_reward": initial_reward,
        "final_reward": final_reward,
        "improvement": improvement,
        "timesteps": timesteps,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    # Auto-detect device
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    print("=" * 60)
    print("PPO Smoke Test — BlueSkyMARLEnv")
    print(f"Device: {device}")
    print(f"Multi-scenario: {args.multi_scenario}")
    print("=" * 60)

    output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp())

    if args.multi_scenario:
        # Multi-scenario training
        results: list[dict[str, Any]] = []
        for cfg in CORE_SCENARIOS:
            result = train_single_scenario(cfg, args.timesteps, args.seed, output_dir)
            results.append(result)

        # Save training curve CSV
        csv_path = output_dir / "training_curve.csv"
        save_training_curve_csv(results, csv_path)
        print(f"\n{'='*60}")
        print(f"Training curve saved to {csv_path}")
        print(f"{'='*60}")

        # Print summary
        print("\nSummary:")
        for r in results:
            status = "SUCCESS" if r["improvement"] > 0 else "WARNING"
            init_r = r["initial_reward"]
            final_r = r["final_reward"]
            imp = r["improvement"]
            print(f"  {r['scenario']}: {init_r:.2f} -> {final_r:.2f} ({imp:+.2f}) [{status}]")

        # Overall result
        all_positive = all(r["improvement"] > 0 for r in results)
        if all_positive:
            print("\nSUCCESS: All scenarios showed learning improvement.")
            return 0
        else:
            print("\nWARNING: Some scenarios did not show improvement.")
            return 1
    else:
        # Single scenario training (original behavior)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Create env
            env = _make_env(tmp_path)

            # Create model
            from stable_baselines3 import PPO
            model = PPO(
                "MultiInputPolicy",
                env,
                n_steps=128,
                batch_size=64,
                n_epochs=4,
                learning_rate=3e-4,
                verbose=0,
                device=device,
                seed=args.seed,
            )

            # Evaluate before training
            initial_reward = _evaluate(model, env, n_episodes=5)
            print(f"\n[Before training] Mean reward: {initial_reward:.2f}")

            # Train
            print("Training for 10,000 timesteps...")
            model.learn(total_timesteps=10_000, callback=[ProgressCallback()])

            # Evaluate after training
            final_reward = _evaluate(model, env, n_episodes=5)
            print(f"[After training]  Mean reward: {final_reward:.2f}")

            env.close()

        # Check result
        improvement = final_reward - initial_reward
        print(f"\nImprovement: {improvement:+.2f}")

        if final_reward > initial_reward:
            print("\nSUCCESS: PPO learned to improve rewards.")
            return 0
        else:
            print("\nWARNING: No improvement detected. This may be due to")
            print("randomness — try running again or increasing timesteps.")
            return 1
    from stable_baselines3 import PPO

    # Auto-detect device
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    print("=" * 60)
    print("PPO Smoke Test — BlueSkyMARLEnv + WaypointNav")
    print(f"Device: {device}")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Create env
        env = _make_env(tmp_path)

        # Create model
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            learning_rate=3e-4,
            verbose=0,
            device=device,
            seed=42,
        )

        # Evaluate before training
        initial_reward = _evaluate(model, env, n_episodes=5)
        print(f"\n[Before training] Mean reward: {initial_reward:.2f}")

        # Train
        print("Training for 10,000 timesteps...")
        model.learn(total_timesteps=10_000, callback=[ProgressCallback()])

        # Evaluate after training
        final_reward = _evaluate(model, env, n_episodes=5)
        print(f"[After training]  Mean reward: {final_reward:.2f}")

        env.close()

    # Check result
    improvement = final_reward - initial_reward
    print(f"\nImprovement: {improvement:+.2f}")

    if final_reward > initial_reward:
        print("\nSUCCESS: PPO learned to improve rewards.")
        return 0
    else:
        print("\nWARNING: No improvement detected. This may be due to")
        print("randomness — try running again or increasing timesteps.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
