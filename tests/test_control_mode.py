"""Tests for scenario control mode interface (control_mode, ego_agent, background_agents)."""

from __future__ import annotations

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.types import ConflictConfig, SpawnConfig


class DummyScenario(BaseScenario):
    """Minimal concrete scenario for testing base class defaults."""

    def setup(self, rng, airspace_bounds):
        return ["AC000", "AC001", "AC002"]

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


class TestControlModeDefaults:
    """Verify BaseScenario default control mode properties."""

    def test_default_control_mode_is_multi_rl(self):
        scenario = DummyScenario()
        assert scenario.control_mode == "MULTI_RL"

    def test_default_ego_agent_is_none(self):
        scenario = DummyScenario()
        assert scenario.ego_agent is None

    def test_default_background_agents_is_empty(self):
        scenario = DummyScenario()
        assert scenario.background_agents == []

    def test_control_mode_is_string(self):
        scenario = DummyScenario()
        assert isinstance(scenario.control_mode, str)


class SingleRLScenario(BaseScenario):
    """Scenario that overrides control mode to SINGLE_RL."""

    @property
    def control_mode(self):
        return "SINGLE_RL"

    @property
    def ego_agent(self):
        return "AC000"

    @property
    def background_agents(self):
        return [f"AC{i:03d}" for i in range(1, 20)]

    def setup(self, rng, airspace_bounds):
        return [f"AC{i:03d}" for i in range(20)]

    def get_spawn_config(self):
        return SpawnConfig(
            altitude_range=(3000, 8000),
            speed_range=(200, 280),
            heading_range=(0, 360),
        )

    def get_conflict_config(self):
        return ConflictConfig(
            nmac_horizontal_nm=4,
            nmac_vertical_ft=500,
            warning_horizontal_nm=8,
            warning_vertical_ft=1000,
        )

    def get_waypoint(self, agent_id):
        return {"lat": 40.0, "lon": 117.0, "alt": 5000.0, "hdg": 0.0}


class TestControlModeSingleRL:
    """Verify SINGLE_RL mode overrides."""

    def test_single_rl_control_mode(self):
        scenario = SingleRLScenario()
        assert scenario.control_mode == "SINGLE_RL"

    def test_single_rl_ego_agent(self):
        scenario = SingleRLScenario()
        assert scenario.ego_agent == "AC000"

    def test_single_rl_background_agents(self):
        scenario = SingleRLScenario()
        bg = scenario.background_agents
        assert len(bg) == 19
        assert "AC000" not in bg
        assert "AC001" in bg
