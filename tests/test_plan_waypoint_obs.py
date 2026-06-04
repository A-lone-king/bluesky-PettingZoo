"""Tests for PlanWaypoint observation support in ObservationManager."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_pettingzoo.envs.scenarios.plan_waypoint import PlanWaypointScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.utils.types import AircraftState
from tests.helpers.env_factory import make_config


@pytest.fixture
def obs_manager():
    config = make_config()
    return ObservationManager(config)


@pytest.fixture
def scenario():
    s = PlanWaypointScenario(seed=42)
    bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
    rng = np.random.RandomState(42)
    s.setup(rng, bounds)
    return s


def _make_state(lat=40.0, lon=117.0, alt=35000.0, hdg=90.0, tas=450.0):
    return AircraftState(
        id="AC000",
        lat=lat,
        lon=lon,
        alt=alt,
        hdg=hdg,
        tas=tas,
        vs=0.0,
    )


class TestPlanWaypointObservation:
    """Verify PlanWaypoint-specific observation features."""

    def test_generate_with_waypoints(self, obs_manager, scenario):
        """Observation generation should accept waypoints parameter."""
        state = _make_state()
        goal = scenario.get_waypoint("AC000")
        result = obs_manager.generate(
            own_state=state,
            other_states=[],
            goal=goal,
            waypoints=scenario._waypoints,
            waypoints_reached=scenario._reached,
        )
        assert "observation" in result

    def test_observation_has_waypoints_key(self, obs_manager, scenario):
        """Observation dict should contain 'waypoints' key."""
        state = _make_state()
        goal = scenario.get_waypoint("AC000")
        result = obs_manager.generate(
            own_state=state,
            other_states=[],
            goal=goal,
            waypoints=scenario._waypoints,
            waypoints_reached=scenario._reached,
        )
        obs = result["observation"]
        assert "waypoints" in obs

    def test_waypoints_features_shape(self, obs_manager, scenario):
        """Waypoints features should have shape (5, 4): distance, cos, sin, reached."""
        state = _make_state()
        goal = scenario.get_waypoint("AC000")
        result = obs_manager.generate(
            own_state=state,
            other_states=[],
            goal=goal,
            waypoints=scenario._waypoints,
            waypoints_reached=scenario._reached,
        )
        wp_obs = result["observation"]["waypoints"]
        assert wp_obs["features"].shape == (5, 4)

    def test_waypoints_mask_shape(self, obs_manager, scenario):
        """Waypoints mask should have shape (5,)."""
        state = _make_state()
        goal = scenario.get_waypoint("AC000")
        result = obs_manager.generate(
            own_state=state,
            other_states=[],
            goal=goal,
            waypoints=scenario._waypoints,
            waypoints_reached=scenario._reached,
        )
        wp_obs = result["observation"]["waypoints"]
        assert wp_obs["mask"].shape == (5,)

    def test_reached_waypoint_has_mask_zero(self, obs_manager, scenario):
        """Reached waypoints should have mask=0."""
        scenario.mark_reached(0)
        state = _make_state()
        goal = scenario.get_waypoint("AC000")
        result = obs_manager.generate(
            own_state=state,
            other_states=[],
            goal=goal,
            waypoints=scenario._waypoints,
            waypoints_reached=scenario._reached,
        )
        mask = result["observation"]["waypoints"]["mask"]
        assert mask[0] == 0  # reached
        assert mask[1] == 1  # not reached
