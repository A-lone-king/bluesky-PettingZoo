"""Tests for priority-based conflict resolution."""

from __future__ import annotations

import numpy as np

from bluesky_pettingzoo.envs.scenarios.horizontal_cr import HorizontalCRScenario
from bluesky_pettingzoo.envs.scenarios.sector_cr import SectorCRScenario
from bluesky_pettingzoo.envs.scenarios.vertical_cr import VerticalCRScenario
from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario
from bluesky_pettingzoo.utils.types import AircraftState


def _make_state(acid: str, alt: float = 35000.0, tas: float = 450.0) -> AircraftState:
    return AircraftState(id=acid, lat=39.25, lon=116.25, alt=alt, hdg=90.0, tas=tas, vs=0.0)


class TestBaseScenarioDefaultPriority:
    """Default get_priority returns 0.0."""

    def test_waypoint_nav_default_priority(self) -> None:
        scenario = WaypointNavScenario(num_aircraft=3, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        scenario.setup(rng, bounds)
        state = _make_state("AC000")
        assert scenario.get_priority("AC000", state) == 0.0


class TestHorizontalCRPriority:
    """HorizontalCR: higher altitude → higher priority."""

    def test_higher_altitude_higher_priority(self) -> None:
        scenario = HorizontalCRScenario(num_aircraft=3, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        scenario.setup(rng, bounds)
        high_alt = _make_state("AC000", alt=37000.0)
        low_alt = _make_state("AC001", alt=29000.0)
        assert scenario.get_priority("AC000", high_alt) > scenario.get_priority("AC001", low_alt)

    def test_priority_normalized(self) -> None:
        """Priority should be in a reasonable range."""
        scenario = HorizontalCRScenario(num_aircraft=3, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        scenario.setup(rng, bounds)
        state = _make_state("AC000", alt=35000.0)
        p = scenario.get_priority("AC000", state)
        assert -1.0 <= p <= 1.0


class TestVerticalCRPriority:
    """VerticalCR: faster speed → higher priority."""

    def test_faster_speed_higher_priority(self) -> None:
        scenario = VerticalCRScenario(num_aircraft=3, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        scenario.setup(rng, bounds)
        fast = _make_state("AC000", tas=500.0)
        slow = _make_state("AC001", tas=400.0)
        assert scenario.get_priority("AC000", fast) > scenario.get_priority("AC001", slow)


class TestSectorCRPriority:
    """SectorCR: closer to waypoint → higher priority."""

    def test_closer_to_waypoint_higher_priority(self) -> None:
        scenario = SectorCRScenario(num_aircraft=3, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        scenario.setup(rng, bounds)
        wp = scenario.get_waypoint("AC000")
        # Aircraft close to waypoint
        close = AircraftState(
            id="AC000",
            lat=wp["lat"],
            lon=wp["lon"],
            alt=35000.0,
            hdg=90.0,
            tas=450.0,
            vs=0.0,
        )
        # Aircraft far from waypoint
        far = AircraftState(
            id="AC001",
            lat=39.0,
            lon=116.0,
            alt=35000.0,
            hdg=90.0,
            tas=450.0,
            vs=0.0,
        )
        assert scenario.get_priority("AC000", close) > scenario.get_priority("AC001", far)
