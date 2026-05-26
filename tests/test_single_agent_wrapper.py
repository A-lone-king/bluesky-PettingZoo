"""Tests for SingleAgentGymWrapper with continuous action space (spec4 F2).

Verify that wrapper works correctly with continuous action space.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario


@pytest.fixture
def continuous_scenario():
    """Create a continuous scenario."""

    class ContinuousScenario(BaseScenario):
        action_space_type = "continuous"

        def setup(self, rng, airspace_bounds):
            return [f"AC{i:03d}" for i in range(3)]

        def get_spawn_config(self):
            from bluesky_pettingzoo.utils.types import SpawnConfig
            return SpawnConfig(
                altitude_range=(30000, 40000),
                speed_range=(400, 500),
                heading_range=(0, 360),
            )

        def get_conflict_config(self):
            from bluesky_pettingzoo.utils.types import ConflictConfig
            return ConflictConfig(
                nmac_horizontal_nm=5,
                nmac_vertical_ft=1000,
                warning_horizontal_nm=10,
                warning_vertical_ft=2000,
            )

        def get_waypoint(self, agent_id):
            return {"lat": 39.5, "lon": 116.5, "alt": 35000, "hdg": 90}

    return ContinuousScenario()


class TestContinuousWrapper:
    """SingleAgentGymWrapper should work with continuous action space."""

    def test_action_space_type(self, continuous_scenario, tmp_path) -> None:
        """Wrapper should expose Box action space for continuous scenario."""
        from bluesky_pettingzoo.wrappers.single_agent import SingleAgentGymWrapper
        from tests.helpers.env_factory import make_env

        env = make_env(tmp_path, scenario=continuous_scenario, initial_count=3)
        wrapper = SingleAgentGymWrapper(env, ego_agent="AC000")

        assert isinstance(wrapper.action_space, gym.spaces.Box)

    def test_action_space_shape(self, continuous_scenario, tmp_path) -> None:
        """Wrapper action space should have shape (3,) for continuous."""
        from bluesky_pettingzoo.wrappers.single_agent import SingleAgentGymWrapper
        from tests.helpers.env_factory import make_env

        env = make_env(tmp_path, scenario=continuous_scenario, initial_count=3)
        wrapper = SingleAgentGymWrapper(env, ego_agent="AC000")

        assert wrapper.action_space.shape == (3,)

    def test_step_with_continuous_action(self, continuous_scenario, tmp_path) -> None:
        """Wrapper should accept continuous action array."""
        from bluesky_pettingzoo.wrappers.single_agent import SingleAgentGymWrapper
        from tests.helpers.env_factory import make_env

        env = make_env(tmp_path, scenario=continuous_scenario, initial_count=3)
        wrapper = SingleAgentGymWrapper(env, ego_agent="AC000")

        obs, info = wrapper.reset(seed=42)
        action = np.array([0.5, 0.0, -0.5], dtype=np.float32)

        obs, reward, terminated, truncated, info = wrapper.step(action)

        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_reset_works(self, continuous_scenario, tmp_path) -> None:
        """Wrapper reset should work with continuous scenario."""
        from bluesky_pettingzoo.wrappers.single_agent import SingleAgentGymWrapper
        from tests.helpers.env_factory import make_env

        env = make_env(tmp_path, scenario=continuous_scenario, initial_count=3)
        wrapper = SingleAgentGymWrapper(env, ego_agent="AC000")

        obs, info = wrapper.reset(seed=42)

        assert obs is not None
        assert isinstance(obs, dict)
