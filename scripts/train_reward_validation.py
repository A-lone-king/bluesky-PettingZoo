"""P0-2 reward validation: train PPO on simplified HorizontalCR scenario (fixed airspace).

Trains PPO on HorizontalCR with 2 aircraft, 50 max steps, 100K timesteps.
Uses action_frequency=3 so each RL step covers 15s sim time (~937 NM at 450 kts).
Records training curve (per-episode rewards at intervals) and evaluates
arrival rate after training.

Success criteria:
  - Training completes without error
  - Reward curve shows upward trend (final reward > initial reward)
  - Arrival rate > 10%

Usage:
    python scripts/train_reward_validation.py
    python scripts/train_reward_validation.py --timesteps 50000
    python scripts/train_reward_validation.py --eval-interval 5000
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.horizontal_cr import HorizontalCRScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.capacity import CapacityPenalty
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.fairness import FairnessReward
from bluesky_pettingzoo.rewards.components.flow_efficiency import FlowEfficiencyReward
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from bluesky_pettingzoo.training.progress import ProgressCallback
from bluesky_pettingzoo.wrappers.single_agent import SingleAgentGymWrapper
from tests.helpers.env_factory import make_config, write_rewards_yaml


def _make_env(
    tmp_path: Path,
    num_aircraft: int = 2,
    max_steps: int = 50,
    seed: int = 42,
) -> SingleAgentGymWrapper:
    """Create a SingleAgentGymWrapper for HorizontalCR.

    With action_frequency=3 and dt=5s, one RL step = 15s sim time.
    50 RL steps covers ~937 NM at 450 kts.
    """
    scenario = HorizontalCRScenario(num_aircraft=num_aircraft, seed=seed)
    config = make_config(
        initial_count=num_aircraft,
        max_steps=max_steps,
        airspace={"sectors": [{"id": "s1", "bounds": [[36.0, 112.0], [42.0, 120.0]]}]},
    )
    config["simulation"]["action_frequency"] = 3
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
    if hasattr(scenario, "get_sectors"):
        calc.register(CapacityPenalty(merged), weight=1.0)
        calc.register(FlowEfficiencyReward(merged), weight=0.2)
        calc.register(FairnessReward(merged), weight=0.1)

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


def _evaluate(
    model: Any, env: SingleAgentGymWrapper, n_episodes: int = 20
) -> tuple[float, float]:
    """Evaluate model and return (mean_reward, arrival_rate).

    An arrival is defined as the episode ending with a positive total reward
    (i.e. the ego agent reached its waypoint).
    """
    rewards: list[float] = []
    arrivals = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
        if total > 0:
            arrivals += 1
    return float(np.mean(rewards)), arrivals / n_episodes


def train_and_validate(
    timesteps: int = 100_000,
    num_aircraft: int = 2,
    max_steps: int = 50,
    eval_interval: int = 10_000,
    seed: int = 42,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run PPO training and validate reward tuning.

    Returns a dict with training results including reward curve and arrival rate.
    """
    from stable_baselines3 import PPO

    output_path = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"P0-2 Reward Validation: HorizontalCR")
    print(f"  Aircraft: {num_aircraft}, Max steps: {max_steps}")
    print(f"  Total timesteps: {timesteps:,}")
    print(f"  Eval interval: {eval_interval:,}")
    print(f"  Seed: {seed}")
    print(f"  Output: {output_path}")
    print(f"=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        env = _make_env(tmp_path, num_aircraft=num_aircraft, max_steps=max_steps, seed=seed)

        # Create PPO model
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            verbose=0,
            device="cpu",
            seed=seed,
        )

        # Evaluate before training
        initial_reward, initial_arrival = _evaluate(model, env, n_episodes=20)
        print(f"\n[Before] Mean reward: {initial_reward:.2f}, Arrival rate: {initial_arrival:.1%}")

        # Train with periodic evaluation
        curve_data: list[dict[str, Any]] = []
        curve_data.append({
            "timestep": 0,
            "mean_reward": initial_reward,
            "arrival_rate": initial_arrival,
        })

        # Train in chunks to record the curve
        remaining = timesteps
        current_t = 0
        while remaining > 0:
            chunk = min(eval_interval, remaining)
            model.learn(
                total_timesteps=chunk,
                callback=[ProgressCallback()],
            )
            current_t += chunk
            remaining -= chunk

            # Evaluate
            mean_r, arrival_r = _evaluate(model, env, n_episodes=20)
            curve_data.append({
                "timestep": current_t,
                "mean_reward": mean_r,
                "arrival_rate": arrival_r,
            })
            print(f"  [{current_t:>8,}/{timesteps:,}] Reward: {mean_r:.2f}, Arrival: {arrival_r:.1%}")

        # Final evaluation with more episodes
        final_reward, final_arrival = _evaluate(model, env, n_episodes=50)
        print(f"\n[Final]  Mean reward: {final_reward:.2f}, Arrival rate: {final_arrival:.1%}")

        # Save model
        model_path = output_path / "ppo_horizontalcr_final.zip"
        model.save(str(model_path))
        print(f"  Model saved to {model_path}")

        env.close()

    # Save training curve CSV
    csv_path = output_path / "training_curve.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestep", "mean_reward", "arrival_rate"])
        writer.writeheader()
        writer.writerows(curve_data)
    print(f"  Training curve saved to {csv_path}")

    # Analyze trend
    rewards_list = [d["mean_reward"] for d in curve_data]
    has_upward_trend = final_reward > initial_reward

    # Calculate trend using linear regression
    x = np.arange(len(rewards_list))
    slope = float(np.polyfit(x, rewards_list, 1)[0])

    # Results
    result = {
        "initial_reward": initial_reward,
        "final_reward": final_reward,
        "improvement": final_reward - initial_reward,
        "has_upward_trend": has_upward_trend,
        "trend_slope": slope,
        "initial_arrival_rate": initial_arrival,
        "final_arrival_rate": final_arrival,
        "arrival_rate_target_met": final_arrival > 0.10,
        "timesteps": timesteps,
        "curve_data": curve_data,
    }

    # Save results JSON
    import json
    results_path = output_path / "validation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        # Convert numpy types for JSON serialization
        serializable = {k: v for k, v in result.items() if k != "curve_data"}
        json.dump(serializable, f, indent=2)
    print(f"  Results saved to {results_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"P0-2 Validation Summary")
    print(f"{'=' * 60}")
    print(f"  Reward: {initial_reward:.2f} -> {final_reward:.2f} ({final_reward - initial_reward:+.2f})")
    print(f"  Trend slope: {slope:.4f} ({'UP' if slope > 0 else 'DOWN'})")
    print(f"  Arrival rate: {final_arrival:.1%} (target: >10%)")

    passed = True
    if has_upward_trend:
        print(f"  [PASS] Reward curve has upward trend")
    else:
        print(f"  [WARN] No upward trend detected (may need more timesteps)")
        passed = False

    if final_arrival > 0.10:
        print(f"  [PASS] Arrival rate > 10%")
    else:
        print(f"  [WARN] Arrival rate < 10% (may need more timesteps)")
        passed = False

    print(f"{'=' * 60}")

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="P0-2 reward validation: PPO on simplified HorizontalCR"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Total training timesteps (default: 100000)",
    )
    parser.add_argument(
        "--num-aircraft", type=int, default=2,
        help="Number of aircraft (default: 2)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=50,
        help="Max steps per episode (default: 50)",
    )
    parser.add_argument(
        "--eval-interval", type=int, default=10_000,
        help="Evaluation interval in timesteps (default: 10000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: temp dir)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    result = train_and_validate(
        timesteps=args.timesteps,
        num_aircraft=args.num_aircraft,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    return 0 if (result["has_upward_trend"] and result["arrival_rate_target_met"]) else 1


if __name__ == "__main__":
    sys.exit(main())
