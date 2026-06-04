"""Tests for PlanWaypoint scenario YAML configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario

SCENARIOS_DIR = Path(__file__).parent.parent / "config" / "scenarios"


class TestPlanWaypointConfig:
    """Verify PlanWaypoint YAML configuration."""

    def test_yaml_file_exists(self):
        path = SCENARIOS_DIR / "plan_waypoint.yaml"
        assert path.exists()

    def test_yaml_has_scenario_key(self):
        path = SCENARIOS_DIR / "plan_waypoint.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["scenario"] == "PlanWaypoint"

    def test_yaml_has_action_space(self):
        path = SCENARIOS_DIR / "plan_waypoint.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "action_space" in data

    def test_yaml_has_control_mode(self):
        path = SCENARIOS_DIR / "plan_waypoint.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "control_mode" in data

    def test_from_config_loads_scenario(self):
        path = SCENARIOS_DIR / "plan_waypoint.yaml"
        scenario = BaseScenario.from_config(path)
        assert scenario.__class__.__name__ == "PlanWaypointScenario"
