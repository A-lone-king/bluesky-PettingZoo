"""Tests for YAML configuration consistency.

Validates that scenario reward_overrides keys match base rewards.yaml keys.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).parent.parent / "config"
REWARDS_YAML = CONFIG_DIR / "rewards.yaml"
SCENARIOS_DIR = CONFIG_DIR / "scenarios"


class TestRewardsYamlCompleteness:
    """Verify rewards.yaml contains all registered reward components."""

    def test_altitude_reward_in_rewards_yaml(self):
        with open(REWARDS_YAML, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "altitude_reward" in data.get("components", {}), (
            "altitude_reward missing from config/rewards.yaml"
        )

    def test_all_registered_components_in_yaml(self):
        from bluesky_pettingzoo.rewards.components import __all__ as registered
        with open(REWARDS_YAML, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        yaml_keys = set(data.get("components", {}).keys())
        # Map class names to expected YAML keys
        class_to_key = {
            "AltitudeReward": "altitude_reward",
            "CapacityPenalty": "capacity",
            "ConflictPenalty": "conflict",
            "DelayPenalty": "delay",
            "DriftPenalty": "drift_penalty",
            "EfficiencyReward": "efficiency",
            "FairnessReward": "fairness",
            "FlowEfficiencyReward": "flow_efficiency",
            "ObstacleIntrusion": "obstacle_intrusion",
            "SmoothnessPenalty": "smoothness",
        }
        for cls_name in registered:
            expected_key = class_to_key.get(cls_name)
            if expected_key:
                assert expected_key in yaml_keys, (
                    f"{cls_name} expects key '{expected_key}' in rewards.yaml, but not found"
                )


class TestScenarioOverrideKeys:
    """Verify scenario reward_overrides keys match base rewards.yaml keys."""

    def _get_base_keys(self) -> set[str]:
        with open(REWARDS_YAML, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return set(data.get("components", {}).keys())

    def _get_scenario_overrides(self, scenario_name: str) -> set[str]:
        path = SCENARIOS_DIR / f"{scenario_name}.yaml"
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return set(data.get("reward_overrides", {}).keys())

    @pytest.mark.parametrize("scenario", [
        "horizontal_cr", "vertical_cr", "sector_cr", "plan_waypoint",
        "descent", "merge", "route_nav", "sector_capacity",
        "static_obstacle", "waypoint_nav",
    ])
    def test_scenario_override_keys_match_base(self, scenario: str):
        base_keys = self._get_base_keys()
        override_keys = self._get_scenario_overrides(scenario)
        invalid = override_keys - base_keys
        assert not invalid, (
            f"{scenario}.yaml has override keys {invalid} not in rewards.yaml. "
            f"Valid keys: {sorted(base_keys)}"
        )
