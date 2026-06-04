"""Tests for BaseScenario.create_intruders() default method."""

from __future__ import annotations

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.types import ConflictConfig, SpawnConfig


class DummyScenario(BaseScenario):
    def setup(self, rng, airspace_bounds):
        return ["AC000"]

    def get_spawn_config(self):
        return SpawnConfig(
            altitude_range=(30000, 40000),
            speed_range=(400, 500),
            heading_range=(0, 360),
        )

    def get_conflict_config(self):
        return ConflictConfig(
            nmac_horizontal_nm=5,
            nmac_vertical_ft=1000,
            warning_horizontal_nm=10,
            warning_vertical_ft=2000,
        )

    def get_waypoint(self, agent_id):
        return {"lat": 40.0, "lon": 117.0, "alt": 35000.0, "hdg": 90.0}


class TestCreateIntruders:
    """Verify create_intruders default behavior."""

    def test_default_returns_empty_list(self):
        scenario = DummyScenario()
        result = scenario.create_intruders(None, None)
        assert result == []

    def test_default_returns_list(self):
        scenario = DummyScenario()
        result = scenario.create_intruders(None, None)
        assert isinstance(result, list)

    def test_method_exists_on_base(self):
        assert hasattr(BaseScenario, "create_intruders")
