"""Tests for PlanWaypointScenario."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_pettingzoo.envs.scenarios.plan_waypoint import PlanWaypointScenario


@pytest.fixture
def scenario():
    return PlanWaypointScenario(seed=42)


@pytest.fixture
def setup_scenario(scenario):
    bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
    rng = np.random.RandomState(42)
    agents = scenario.setup(rng, bounds)
    return scenario, agents, rng


class TestPlanWaypointSetup:
    """Verify scenario setup returns correct agent count and waypoints."""

    def test_setup_returns_one_agent(self, setup_scenario):
        _, agents, _ = setup_scenario
        assert len(agents) == 1

    def test_agent_id_is_ac000(self, setup_scenario):
        _, agents, _ = setup_scenario
        assert agents[0] == "AC000"

    def test_has_five_waypoints(self, setup_scenario):
        scenario, _, _ = setup_scenario
        wp = scenario.get_waypoint("AC000")
        assert isinstance(wp, dict)
        # PlanWaypoint stores waypoints as a list internally
        assert len(scenario._waypoints) == 5

    def test_waypoint_has_lat_lon(self, setup_scenario):
        scenario, _, _ = setup_scenario
        for wp in scenario._waypoints:
            assert "lat" in wp
            assert "lon" in wp

    def test_action_dimensions_heading_only(self, setup_scenario):
        scenario, _, _ = setup_scenario
        assert scenario.action_dimensions == [0]


class TestPlanWaypointArrival:
    """Verify waypoint arrival detection and clearing."""

    def test_mark_reached(self, setup_scenario):
        scenario, _, _ = setup_scenario
        # Mark first waypoint as reached
        scenario.mark_reached(0)
        assert scenario._reached[0] is True

    def test_current_waypoint(self, setup_scenario):
        scenario, _, _ = setup_scenario
        # First waypoint should be current
        wp = scenario.get_waypoint("AC000")
        assert wp == scenario._waypoints[0]

    def test_get_reached_count(self, setup_scenario):
        scenario, _, _ = setup_scenario
        assert scenario.get_reached_count() == 0
        scenario.mark_reached(0)
        assert scenario.get_reached_count() == 1

    def test_all_reached(self, setup_scenario):
        scenario, _, _ = setup_scenario
        assert not scenario.all_reached()
        for i in range(5):
            scenario.mark_reached(i)
        assert scenario.all_reached()
