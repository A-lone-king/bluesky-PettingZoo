"""Smoke test: verify PPO can learn in BlueSkyMARLEnv.

Uses BlueSkyWrapper + WaypointNavScenario (1 aircraft, no conflicts).
The agent must learn to turn toward its waypoint to collect arrival rewards.

Success criteria:
  - Training completes without error over 10k timesteps
  - Mean episode reward improves from first to last evaluation
  - Final mean reward > initial mean reward (learning signal present)

Usage:
    python scripts/train_smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Ensure project root is on the path for test helpers
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from bluesky_pettingzoo.wrappers.single_agent import SingleAgentGymWrapper

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from tests.helpers.env_factory import make_config, write_rewards_yaml


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


def _evaluate(model, env, n_episodes: int = 5) -> float:
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


def main() -> int:
    from stable_baselines3 import PPO

    print("=" * 60)
    print("PPO Smoke Test — BlueSkyMARLEnv + WaypointNav")
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
            device="cpu",
            seed=42,
        )

        # Evaluate before training
        initial_reward = _evaluate(model, env, n_episodes=5)
        print(f"\n[Before training] Mean reward: {initial_reward:.2f}")

        # Train
        print("Training for 10,000 timesteps...")
        model.learn(total_timesteps=10_000)

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
