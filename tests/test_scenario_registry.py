"""Tests for scenario registry — all 11 scenarios registered."""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.envs.scenarios.base import _SCENARIO_REGISTRY

EXPECTED_SCENARIOS = {
    "HorizontalCR": "HorizontalCRScenario",
    "VerticalCR": "VerticalCRScenario",
    "SectorCR": "SectorCRScenario",
    "PlanWaypoint": "PlanWaypointScenario",
    "Descent": "DescentScenario",
    "Merge": "MergeScenario",
    "RouteNav": "RouteNavScenario",
    "SectorCapacity": "SectorCapacityScenario",
    "StaticObstacle": "StaticObstacleScenario",
    "StarApproach": "StarApproachScenario",
    "WaypointNav": "WaypointNavScenario",
}


class TestScenarioRegistry:
    """Verify all 11 scenarios are registered."""

    @pytest.mark.parametrize("name", list(EXPECTED_SCENARIOS.keys()))
    def test_scenario_in_registry(self, name: str):
        assert name in _SCENARIO_REGISTRY, f"{name} missing from _SCENARIO_REGISTRY"

    @pytest.mark.parametrize("name,cls_name", list(EXPECTED_SCENARIOS.items()))
    def test_scenario_class_name(self, name: str, cls_name: str):
        assert _SCENARIO_REGISTRY[name] == cls_name

    def test_registry_has_exactly_11_entries(self):
        assert len(_SCENARIO_REGISTRY) == 11


class TestScenarioImportable:
    """All registered scenarios must be importable."""

    @pytest.mark.parametrize("name,cls_name", list(EXPECTED_SCENARIOS.items()))
    def test_scenario_class_importable(self, name: str, cls_name: str):
        import bluesky_pettingzoo.envs.scenarios as scenarios_mod

        assert hasattr(scenarios_mod, cls_name), f"{cls_name} not importable from scenarios module"

    @pytest.mark.parametrize("name,cls_name", list(EXPECTED_SCENARIOS.items()))
    def test_scenario_in_all(self, name: str, cls_name: str):
        from bluesky_pettingzoo.envs.scenarios import __all__

        assert cls_name in __all__, f"{cls_name} missing from __all__"
