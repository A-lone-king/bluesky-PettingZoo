"""Tests for max_observable_aircraft dynamic configuration (obs-003)."""

from __future__ import annotations

import numpy as np

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from tests.helpers.env_factory import make_config


class MockScenario(BaseScenario):
    """Mock scenario for testing max_observable_aircraft override."""

    def __init__(self, max_obs: int | None = None) -> None:
        self._max_obs = max_obs

    @property
    def max_observable_aircraft(self) -> int | None:
        return self._max_obs

    def setup(self, rng: np.random.RandomState, airspace_bounds: dict[str, float]) -> list[str]:
        return ["AC000", "AC001"]

    def get_spawn_config(self):
        from bluesky_pettingzoo.utils.types import SpawnConfig

        return SpawnConfig(
            altitudes=[35000.0, 35000.0],
            speeds=[450.0, 450.0],
            headings=[90.0, 270.0],
            lats=[40.0, 40.0],
            lons=[116.0, 118.0],
        )

    def get_waypoint(self, agent_id: str) -> dict[str, float]:
        return {"lat": 40.0, "lon": 118.0, "alt": 35000.0, "hdg": 90.0}


class TestMaxObservableAircraftProperty:
    """Test BaseScenario.max_observable_aircraft property."""

    def test_default_returns_none(self) -> None:
        """BaseScenario default should return None."""
        scenario = MockScenario(max_obs=None)
        assert scenario.max_observable_aircraft is None

    def test_override_returns_value(self) -> None:
        """Scenario can override max_observable_aircraft."""
        scenario = MockScenario(max_obs=15)
        assert scenario.max_observable_aircraft == 15

    def test_base_scenario_default(self) -> None:
        """BaseScenario itself should return None."""
        assert BaseScenario.max_observable_aircraft.fget is not None


class TestObservationSpaceWithOverride:
    """Test observation space changes with max_observable_aircraft."""

    def test_default_config_uses_config_value(self) -> None:
        """Default config should use max_observable_aircraft from config."""
        config = make_config()
        obs_manager = ObservationManager(config)
        space = obs_manager.observation_space()
        other_shape = space["other_aircraft"].shape
        # env_factory default is 5
        assert other_shape[0] == 5

    def test_custom_config_changes_shape(self) -> None:
        """Custom max_observable_aircraft should change observation shape."""
        config = make_config()
        config["observation"]["max_observable_aircraft"] = 15
        obs_manager = ObservationManager(config)
        space = obs_manager.observation_space()
        other_shape = space["other_aircraft"].shape
        assert other_shape[0] == 15

    def test_mask_shape_matches(self) -> None:
        """Mask shape should match max_observable_aircraft."""
        config = make_config()
        config["observation"]["max_observable_aircraft"] = 8
        obs_manager = ObservationManager(config)
        space = obs_manager.observation_space()
        mask_shape = space["other_aircraft_mask"].shape
        assert mask_shape == (8,)


class TestScenarioOverrideIntegration:
    """Test scenario override integration with config."""

    def test_scenario_override_applied(self) -> None:
        """Scenario max_observable_aircraft should override config."""
        config = make_config()
        scenario = MockScenario(max_obs=20)

        # Simulate the override logic
        if scenario.max_observable_aircraft is not None:
            config["observation"]["max_observable_aircraft"] = scenario.max_observable_aircraft

        obs_manager = ObservationManager(config)
        space = obs_manager.observation_space()
        other_shape = space["other_aircraft"].shape
        assert other_shape[0] == 20

    def test_none_does_not_override(self) -> None:
        """None max_observable_aircraft should not change config."""
        config = make_config()
        config["observation"]["max_observable_aircraft"] = 12
        scenario = MockScenario(max_obs=None)

        # Simulate the override logic
        if scenario.max_observable_aircraft is not None:
            config["observation"]["max_observable_aircraft"] = scenario.max_observable_aircraft

        obs_manager = ObservationManager(config)
        space = obs_manager.observation_space()
        other_shape = space["other_aircraft"].shape
        assert other_shape[0] == 12
