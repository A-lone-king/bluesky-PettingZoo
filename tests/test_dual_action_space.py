"""Tests for dual action space (spec4 F2).

Verify discrete→MultiDiscrete and continuous→Box action spaces.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario


@pytest.fixture
def discrete_scenario():
    """Create a discrete scenario."""

    class DiscreteScenario(BaseScenario):
        def setup(self, rng, airspace_bounds):
            return []

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

    return DiscreteScenario()


@pytest.fixture
def continuous_scenario():
    """Create a continuous scenario."""

    class ContinuousScenario(BaseScenario):
        action_space_type = "continuous"

        def setup(self, rng, airspace_bounds):
            return []

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


class TestDiscreteActionSpace:
    """Discrete scenario should use MultiDiscrete action space."""

    def test_discrete_type(self, discrete_scenario, tmp_path) -> None:
        """Discrete scenario should have MultiDiscrete action space."""
        from tests.helpers.env_factory import make_env

        env = make_env(tmp_path, scenario=discrete_scenario, initial_count=3)

        action_space = env.action_space("AC000")
        assert isinstance(action_space, gym.spaces.MultiDiscrete)

    def test_discrete_dimensions(self, discrete_scenario, tmp_path) -> None:
        """Discrete action space should have 3 dimensions."""
        from tests.helpers.env_factory import make_env

        env = make_env(tmp_path, scenario=discrete_scenario, initial_count=3)

        action_space = env.action_space("AC000")
        assert action_space.nvec.tolist() == [5, 5, 5]


class TestContinuousActionSpace:
    """Continuous scenario should use Box action space."""

    def test_continuous_type(self, continuous_scenario, tmp_path) -> None:
        """Continuous scenario should have Box action space."""
        from tests.helpers.env_factory import make_env

        env = make_env(tmp_path, scenario=continuous_scenario, initial_count=3)

        action_space = env.action_space("AC000")
        assert isinstance(action_space, gym.spaces.Box)

    def test_continuous_range(self, continuous_scenario, tmp_path) -> None:
        """Continuous action space should be Box(-1, 1)."""
        from tests.helpers.env_factory import make_env

        env = make_env(tmp_path, scenario=continuous_scenario, initial_count=3)

        action_space = env.action_space("AC000")
        assert np.all(action_space.low == -1.0)
        assert np.all(action_space.high == 1.0)
        assert action_space.shape == (3,)
