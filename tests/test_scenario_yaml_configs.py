"""Tests for scenario YAML configuration files."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).parent.parent / "config" / "scenarios"

EXPECTED_SCENARIOS = [
    "horizontal_cr",
    "vertical_cr",
    "sector_cr",
    "plan_waypoint",
    "descent",
    "merge",
    "route_nav",
    "sector_capacity",
    "static_obstacle",
    "waypoint_nav",
]


class TestScenarioYamlConfigs:
    """Verify all scenario YAML configs exist and are valid."""

    @pytest.mark.parametrize("name", EXPECTED_SCENARIOS)
    def test_config_file_exists(self, name: str):
        config_path = CONFIG_DIR / f"{name}.yaml"
        assert config_path.exists(), f"Missing config: {config_path}"

    @pytest.mark.parametrize("name", EXPECTED_SCENARIOS)
    def test_config_has_scenario_key(self, name: str):
        config_path = CONFIG_DIR / f"{name}.yaml"
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "scenario" in data, f"{name}.yaml missing 'scenario' key"

    @pytest.mark.parametrize("name", EXPECTED_SCENARIOS)
    def test_config_scenario_registered(self, name: str):
        from bluesky_pettingzoo.envs.scenarios.base import _SCENARIO_REGISTRY

        config_path = CONFIG_DIR / f"{name}.yaml"
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["scenario"] in _SCENARIO_REGISTRY, (
            f"{name}.yaml scenario '{data['scenario']}' not in registry"
        )
