"""Integration tests for StaticObstacleScenario.

Tests the full lifecycle: reset → step → obstacle detection → termination.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.static_obstacle import StaticObstacleScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.obstacle_intrusion import ObstacleIntrusion
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from tests.helpers.env_factory import make_config as _make_config
from tests.helpers.env_factory import write_rewards_yaml as _write_rewards_yaml


def _make_env_with_obstacles(
    env_config: dict[str, Any],
    scenario: StaticObstacleScenario,
) -> BlueSkyMARLEnv:
    """Create a BlueSkyMARLEnv with a StaticObstacleScenario and ObstacleIntrusion."""
    wrapper = BlueSkyWrapper(env_config)
    obs_manager = ObservationManager(env_config)
    action_translator = ActionTranslator(env_config)

    rewards_path = env_config["_rewards_yaml"]
    with open(rewards_path, encoding="utf-8") as f:
        rewards_cfg = yaml.safe_load(f)
    merged = {**env_config, **rewards_cfg}

    calc = RewardCalculator()
    calc.register(ConflictPenalty(merged), weight=1.0)
    calc.register(SmoothnessPenalty(merged), weight=0.5)
    calc.register(EfficiencyReward(merged), weight=0.3)
    obs_comp = ObstacleIntrusion()
    obs_comp.set_obstacles(scenario.get_obstacles())
    calc.register(obs_comp, weight=1.0)

    return BlueSkyMARLEnv(
        config=env_config,
        wrapper=wrapper,
        observation_manager=obs_manager,
        action_translator=action_translator,
        reward_calculator=calc,
        rewards_config=rewards_cfg,
        scenario=scenario,
    )


class TestStaticObstacleSetup:
    """Scenario initialization should return correct agent count."""

    def test_setup_returns_single_agent(self, tmp_path: Path) -> None:
        scenario = StaticObstacleScenario(num_aircraft=1, num_obstacles=5, seed=42)
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=1, max_steps=10, observation={"max_obstacles": 5})
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        env = _make_env_with_obstacles(config, scenario)
        obs, infos = env.reset(seed=42)

        assert len(env.agents) == 1
        env.close()

    def test_setup_returns_multiple_agents(self, tmp_path: Path) -> None:
        scenario = StaticObstacleScenario(num_aircraft=3, num_obstacles=5, seed=42)
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=3, max_steps=10, observation={"max_obstacles": 5})
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        env = _make_env_with_obstacles(config, scenario)
        obs, infos = env.reset(seed=42)

        assert len(env.agents) == 3
        env.close()


class TestStaticObstacleObservation:
    """Observations should include obstacles field."""

    def test_observation_has_obstacles_key(self, tmp_path: Path) -> None:
        scenario = StaticObstacleScenario(num_aircraft=1, num_obstacles=5, seed=42)
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=1, max_steps=10, observation={"max_obstacles": 5})
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        env = _make_env_with_obstacles(config, scenario)
        obs, infos = env.reset(seed=42)

        for agent_id in env.agents:
            assert "obstacles" in obs[agent_id]
            assert "position" in obs[agent_id]["obstacles"]
            assert "mask" in obs[agent_id]["obstacles"]

        env.close()

    def test_obstacles_mask_has_real_entries(self, tmp_path: Path) -> None:
        scenario = StaticObstacleScenario(num_aircraft=1, num_obstacles=5, seed=42)
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=1, max_steps=10, observation={"max_obstacles": 5})
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        env = _make_env_with_obstacles(config, scenario)
        obs, infos = env.reset(seed=42)

        mask = obs["AC000"]["obstacles"]["mask"]
        assert sum(mask) == 5  # 5 real obstacles

        env.close()


class TestStaticObstacleFullEpisode:
    """Full episode should run without errors."""

    def test_full_episode_completes(self, tmp_path: Path) -> None:
        scenario = StaticObstacleScenario(num_aircraft=1, num_obstacles=5, seed=42)
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=1, max_steps=20, observation={"max_obstacles": 5})
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        env = _make_env_with_obstacles(config, scenario)
        obs, infos = env.reset(seed=42)

        total_reward = 0.0
        for _ in range(20):
            actions = {a: [2, 2, 2] for a in env.agents}
            if not actions:
                break
            obs, rewards, terminations, truncations, infos = env.step(actions)
            total_reward += sum(rewards.values())
            if not env.agents:
                break

        assert np.isfinite(total_reward)
        env.close()


class TestStaticObstacleObservationSpace:
    """Observation space should include obstacles when max_obstacles > 0."""

    def test_observation_space_has_obstacles(self, tmp_path: Path) -> None:
        scenario = StaticObstacleScenario(num_aircraft=1, num_obstacles=5, seed=42)
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=1, max_steps=10, observation={"max_obstacles": 5})
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        env = _make_env_with_obstacles(config, scenario)
        space = env.observation_space("AC000")

        assert "obstacles" in space.spaces
        env.close()

    def test_observation_space_no_obstacles_when_zero(self, tmp_path: Path) -> None:
        scenario = StaticObstacleScenario(num_aircraft=1, num_obstacles=5, seed=42)
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=1, max_steps=10, observation={"max_obstacles": 0})
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        env = _make_env_with_obstacles(config, scenario)
        space = env.observation_space("AC000")

        assert "obstacles" not in space.spaces
        env.close()
