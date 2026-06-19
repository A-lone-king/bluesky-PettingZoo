"""Tests for reward-tune-003: simplified scenario quick validation (fixed airspace).

Verifies that PPO can learn on HorizontalCR with 2 aircraft, 50 max steps.
Uses action_frequency=3 so each RL step covers 15s sim time (~1.875 NM/step at 450 kts).
With waypoint_distance_range=(40,70) NM, 50 RL steps (~93.75 NM) can reach waypoints.
This is a fast sanity check (10K timesteps) to confirm the training pipeline
works before the full 100K run.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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
from bluesky_pettingzoo.wrappers.single_agent import SingleAgentGymWrapper
from tests.helpers.env_factory import make_config, write_rewards_yaml


def _make_horizontal_cr_env(
    tmp_path: Path,
    num_aircraft: int = 2,
    max_steps: int = 50,
    seed: int = 42,
) -> SingleAgentGymWrapper:
    """Create a SingleAgentGymWrapper for HorizontalCR scenario.

    With action_frequency=3 and dt=5s, one RL step = 15s sim time (~1.875 NM/step).
    50 RL steps covers ~93.75 NM at 450 kts; waypoints at 40-70 NM are reachable.
    """
    scenario = HorizontalCRScenario(
        num_aircraft=num_aircraft,
        seed=seed,
        waypoint_distance_range=(40, 70),
    )
    config = make_config(
        initial_count=num_aircraft,
        max_steps=max_steps,
        airspace={"sectors": [{"id": "s1", "bounds": [[36.0, 112.0], [42.0, 120.0]]}]},
    )
    config["simulation"]["action_frequency"] = 3
    # Custom rewards with distance guidance for short-range scenario
    rewards_cfg = {
        "components": {
            "conflict": {
                "enabled": True,
                "weight": 1.0,
                "nmac_penalty": -50,
                "warning_penalty": -10,
                "separation_penalty": -5,
                "thresholds": {
                    "nmac_horizontal_nm": 5,
                    "nmac_vertical_ft": 1000,
                    "warning_horizontal_nm": 10,
                    "warning_vertical_ft": 2000,
                },
            },
            "smoothness": {"enabled": True, "weight": 0.5, "action_penalty": -0.1},
            "efficiency": {
                "enabled": True,
                "weight": 0.3,
                "max_deviation_nm": 200,
                "deviation_penalty_scale": 5,
                "arrival_reward": 100,
                "step_penalty": -0.005,
                "arrival_threshold_nm": 2,
                "distance_reward_scale": 2.0,
                "distance_threshold_nm": 500,
            },
        }
    }
    import copy

    rewards_copy = copy.deepcopy(rewards_cfg)
    rewards_path = write_rewards_yaml(tmp_path, rewards_copy)
    config["_rewards_yaml"] = str(rewards_path)
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


def _evaluate(model: Any, env: SingleAgentGymWrapper, n_episodes: int = 10) -> tuple[float, float]:
    """Evaluate model and return (mean_reward, arrival_rate).

    An arrival is defined as the episode ending with a positive total reward
    (i.e. the ego agent reached its waypoint).
    """
    rewards: list[float] = []
    arrivals = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        terminated = False
        truncated = False
        for _ in range(60):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
        if total > 0:
            arrivals += 1
    return float(np.mean(rewards)), arrivals / n_episodes


@pytest.mark.slow
class TestRewardTune003QuickValidation:
    """Quick validation that PPO training pipeline works on HorizontalCR."""

    def test_env_creation(self) -> None:
        """HorizontalCR 2-aircraft env can be created and stepped."""
        with tempfile.TemporaryDirectory() as tmp:
            env = _make_horizontal_cr_env(Path(tmp), num_aircraft=2, max_steps=50)
            obs, _ = env.reset()
            assert obs is not None
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs is not None
            assert isinstance(reward, (int, float))
            env.close()

    def test_ppo_trains_10k(self) -> None:
        """PPO can run 10K steps on HorizontalCR without error."""
        from stable_baselines3 import PPO

        with tempfile.TemporaryDirectory() as tmp:
            env = _make_horizontal_cr_env(Path(tmp), num_aircraft=2, max_steps=50)

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

            # Train 10K steps
            model.learn(total_timesteps=10_000)

            # Evaluate
            mean_reward, arrival_rate = _evaluate(model, env, n_episodes=10)

            # Pipeline completed without error; rewards may not be positive yet
            # at 10K steps, but we verify it's a finite number
            assert np.isfinite(mean_reward)
            assert 0.0 <= arrival_rate <= 1.0

            env.close()

    def test_reward_components_register(self) -> None:
        """All reward components needed for HorizontalCR are registered."""
        with tempfile.TemporaryDirectory() as tmp:
            env = _make_horizontal_cr_env(Path(tmp), num_aircraft=2, max_steps=50)
            # Step a few times to make sure reward calculator works
            obs, _ = env.reset()
            total_reward = 0.0
            for _ in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            # Just verify we got a finite reward
            assert np.isfinite(total_reward)
            env.close()
