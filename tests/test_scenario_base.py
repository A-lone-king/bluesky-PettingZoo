"""Tests for BaseScenario abstract base class (T-V07)."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.types import (
    AircraftState,
    ConflictConfig,
    SpawnConfig,
)

# ---------------------------------------------------------------------------
# Concrete test implementation
# ---------------------------------------------------------------------------


class DummyScenario(BaseScenario):
    """Minimal concrete implementation for testing the base class."""

    def __init__(self) -> None:
        self._agents: list[str] = []
        self._waypoints: dict[str, dict[str, float]] = {}
        self._reset_called = False
        self._update_calls: list[int] = []

    def setup(
        self,
        rng: np.random.RandomState,
        airspace_bounds: dict[str, float],
    ) -> list[str]:
        self._agents = ["AC000", "AC001", "AC002"]
        self._waypoints = {
            "AC000": {"lat": 39.5, "lon": 116.5, "alt": 35000, "hdg": 90.0},
            "AC001": {"lat": 39.0, "lon": 116.0, "alt": 33000, "hdg": 270.0},
            "AC002": {"lat": 39.3, "lon": 116.3, "alt": 34000, "hdg": 180.0},
        }
        return list(self._agents)

    def get_spawn_config(self) -> SpawnConfig:
        return SpawnConfig(
            altitude_range=(29000, 37000),
            speed_range=(400, 500),
            heading_range=(0, 360),
        )

    def get_conflict_config(self) -> ConflictConfig:
        return ConflictConfig(
            nmac_horizontal_nm=5.0,
            nmac_vertical_ft=1000.0,
            warning_horizontal_nm=10.0,
            warning_vertical_ft=2000.0,
        )

    def get_waypoint(self, agent_id: str) -> dict[str, float]:
        return self._waypoints[agent_id]


# ===========================================================================
# T-V07 tests
# ===========================================================================


class TestBaseScenarioIsAbstract:
    """BaseScenario cannot be instantiated directly."""

    def test_base_scenario_is_abstract(self) -> None:
        """Instantiating BaseScenario directly raises TypeError."""
        with pytest.raises(TypeError):
            BaseScenario()  # type: ignore[abstract]


class TestSetupReturnsAgentIds:
    """setup() should return a list of agent ID strings."""

    def test_setup_returns_agent_ids(self) -> None:
        """Concrete scenario's setup() returns agent ID list."""
        scenario = DummyScenario()
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        agents = scenario.setup(rng, bounds)
        assert isinstance(agents, list)
        assert len(agents) == 3
        assert all(isinstance(a, str) for a in agents)


class TestGetSpawnConfig:
    """get_spawn_config() should return a SpawnConfig."""

    def test_get_spawn_config(self) -> None:
        """Concrete scenario returns SpawnConfig with valid ranges."""
        scenario = DummyScenario()
        cfg = scenario.get_spawn_config()
        assert isinstance(cfg, SpawnConfig)
        assert cfg.altitude_range[0] < cfg.altitude_range[1]
        assert cfg.speed_range[0] < cfg.speed_range[1]


class TestGetConflictConfig:
    """get_conflict_config() should return a ConflictConfig."""

    def test_get_conflict_config(self) -> None:
        """Concrete scenario returns ConflictConfig with thresholds."""
        scenario = DummyScenario()
        cfg = scenario.get_conflict_config()
        assert isinstance(cfg, ConflictConfig)
        assert cfg.nmac_horizontal_nm > 0
        assert cfg.warning_horizontal_nm > cfg.nmac_horizontal_nm


class TestShouldTruncate:
    """should_truncate() should correctly identify out-of-bounds aircraft."""

    def test_should_truncate_inside(self) -> None:
        """Aircraft inside airspace should not be truncated."""
        scenario = DummyScenario()
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        state = AircraftState(id="AC000", lat=39.25, lon=116.25, alt=35000, hdg=90, tas=450, vs=0)
        assert scenario.should_truncate("AC000", state, bounds) is False

    def test_should_truncate_outside_lat(self) -> None:
        """Aircraft outside latitude bounds should be truncated."""
        scenario = DummyScenario()
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        state = AircraftState(id="AC000", lat=39.6, lon=116.25, alt=35000, hdg=90, tas=450, vs=0)
        assert scenario.should_truncate("AC000", state, bounds) is True

    def test_should_truncate_outside_lon(self) -> None:
        """Aircraft outside longitude bounds should be truncated."""
        scenario = DummyScenario()
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        state = AircraftState(id="AC000", lat=39.25, lon=115.9, alt=35000, hdg=90, tas=450, vs=0)
        assert scenario.should_truncate("AC000", state, bounds) is True


class TestGetWaypoint:
    """get_waypoint() should return the assigned waypoint for an agent."""

    def test_get_waypoint(self) -> None:
        """Concrete scenario returns correct waypoint per agent."""
        scenario = DummyScenario()
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        scenario.setup(rng, bounds)

        wp = scenario.get_waypoint("AC000")
        assert isinstance(wp, dict)
        assert wp["lat"] == pytest.approx(39.5)
        assert wp["lon"] == pytest.approx(116.5)

    def test_get_waypoint_different_agents(self) -> None:
        """Different agents have different waypoints."""
        scenario = DummyScenario()
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        scenario.setup(rng, bounds)

        wp0 = scenario.get_waypoint("AC000")
        wp1 = scenario.get_waypoint("AC001")
        assert wp0["lat"] != wp1["lat"] or wp0["lon"] != wp1["lon"]


class TestUpdateDefaultNoop:
    """update() default implementation should return empty list."""

    def test_update_default_noop(self) -> None:
        """BaseScenario.update() returns empty list by default."""
        scenario = DummyScenario()
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        scenario.setup(rng, bounds)

        # update is inherited from BaseScenario (DummyScenario doesn't override it)
        result = scenario.update(1, {})
        assert result == []


class TestResetDefaultNoop:
    """reset() default implementation should not raise."""

    def test_reset_default_noop(self) -> None:
        """BaseScenario.reset() does nothing by default."""
        scenario = DummyScenario()
        rng = np.random.RandomState(42)
        # reset is inherited from BaseScenario (DummyScenario doesn't override it)
        scenario.reset(rng)  # Should not raise


class TestGetInitialPositionsDefault:
    """get_initial_positions() should return None by default."""

    def test_get_initial_positions_returns_none(self) -> None:
        """BaseScenario.get_initial_positions() returns None by default."""
        scenario = DummyScenario()
        assert scenario.get_initial_positions() is None

    def test_get_initial_positions_returns_none_after_setup(self) -> None:
        """BaseScenario.get_initial_positions() returns None even after setup()."""
        scenario = DummyScenario()
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        scenario.setup(rng, bounds)
        assert scenario.get_initial_positions() is None
