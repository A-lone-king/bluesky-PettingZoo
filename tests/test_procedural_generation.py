"""Tests for procedural scene generation (obs-procgen-001).

Verifies that scenarios support dynamic aircraft count randomization
via num_aircraft_range and reset(rng).
"""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_pettingzoo.envs.scenarios.descent import DescentScenario
from bluesky_pettingzoo.envs.scenarios.horizontal_cr import HorizontalCRScenario
from bluesky_pettingzoo.envs.scenarios.merge import MergeScenario
from bluesky_pettingzoo.envs.scenarios.plan_waypoint import PlanWaypointScenario
from bluesky_pettingzoo.envs.scenarios.route_nav import RouteNavScenario
from bluesky_pettingzoo.envs.scenarios.sector_capacity import SectorCapacityScenario
from bluesky_pettingzoo.envs.scenarios.sector_cr import SectorCRScenario
from bluesky_pettingzoo.envs.scenarios.static_obstacle import StaticObstacleScenario
from bluesky_pettingzoo.envs.scenarios.vertical_cr import VerticalCRScenario
from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario

_BOUNDS = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}


class TestNumAircraftRange:
    """Scenarios should expose num_aircraft_range property."""

    @pytest.mark.parametrize(
        "scenario_cls, default_n",
        [
            (HorizontalCRScenario, 5),
            (VerticalCRScenario, 5),
            (SectorCRScenario, 5),
            (WaypointNavScenario, 3),
            (RouteNavScenario, 3),
            (SectorCapacityScenario, 6),
            (DescentScenario, 3),
            (MergeScenario, 20),
            (StaticObstacleScenario, 1),
            (PlanWaypointScenario, 1),
        ],
    )
    def test_default_range_is_none(
        self, scenario_cls: type, default_n: int
    ) -> None:
        """Without num_aircraft_range, property returns None."""
        scenario = scenario_cls(num_aircraft=default_n)
        assert scenario.num_aircraft_range is None

    def test_range_set_when_configured(self) -> None:
        """num_aircraft_range returns configured tuple."""
        scenario = HorizontalCRScenario(num_aircraft=3, num_aircraft_range=(2, 8))
        assert scenario.num_aircraft_range == (2, 8)


class TestProceduralReset:
    """reset(rng) should randomize aircraft count when range is set."""

    @pytest.mark.parametrize(
        "scenario_cls",
        [
            HorizontalCRScenario,
            VerticalCRScenario,
            SectorCRScenario,
            WaypointNavScenario,
            RouteNavScenario,
            SectorCapacityScenario,
            DescentScenario,
            MergeScenario,
            StaticObstacleScenario,
            PlanWaypointScenario,
        ],
    )
    def test_reset_without_range_does_not_change_count(self, scenario_cls: type) -> None:
        """Without num_aircraft_range, reset() does not change aircraft count."""
        scenario = scenario_cls(num_aircraft=5)
        rng = np.random.RandomState(42)
        scenario.reset(rng)
        assert scenario._num_aircraft == 5

    @pytest.mark.parametrize(
        "scenario_cls, lo, hi",
        [
            (HorizontalCRScenario, 2, 8),
            (VerticalCRScenario, 2, 8),
            (SectorCRScenario, 2, 8),
            (WaypointNavScenario, 2, 6),
            (RouteNavScenario, 2, 6),
            (SectorCapacityScenario, 3, 10),
            (DescentScenario, 2, 5),
            (MergeScenario, 10, 30),
            (StaticObstacleScenario, 1, 3),
            (PlanWaypointScenario, 1, 3),
        ],
    )
    def test_reset_randomizes_aircraft_count(
        self, scenario_cls: type, lo: int, hi: int
    ) -> None:
        """With num_aircraft_range, reset() samples new count in [lo, hi]."""
        scenario = scenario_cls(num_aircraft=5, num_aircraft_range=(lo, hi))
        counts = set()
        for seed in range(50):
            rng = np.random.RandomState(seed)
            scenario.reset(rng)
            counts.add(scenario._num_aircraft)
        # With 50 seeds and a range of at least 2, we should see >1 distinct count
        assert len(counts) > 1, f"Expected varied counts, got {counts}"
        # All counts should be within range
        assert all(lo <= c <= hi for c in counts)


class TestProceduralSetup:
    """setup() should work correctly after reset() with new aircraft count."""

    def test_horizontal_cr_setup_after_reset(self) -> None:
        """HorizontalCR generates correct agents after procedural reset."""
        scenario = HorizontalCRScenario(num_aircraft=5, num_aircraft_range=(3, 7))
        rng = np.random.RandomState(42)
        scenario.reset(rng)
        agents = scenario.setup(rng, _BOUNDS)
        assert len(agents) == scenario._num_aircraft
        assert 3 <= len(agents) <= 7

    def test_sector_cr_setup_after_reset(self) -> None:
        """SectorCR generates polygon and agents after procedural reset."""
        scenario = SectorCRScenario(num_aircraft=5, num_aircraft_range=(3, 6))
        rng = np.random.RandomState(42)
        scenario.reset(rng)
        agents = scenario.setup(rng, _BOUNDS)
        assert len(agents) == scenario._num_aircraft
        assert len(scenario._polygon) >= 3

    def test_static_obstacle_setup_after_reset(self) -> None:
        """StaticObstacle generates obstacles after procedural reset."""
        scenario = StaticObstacleScenario(
            num_aircraft=1, num_aircraft_range=(1, 3), num_obstacles=5
        )
        rng = np.random.RandomState(42)
        scenario.reset(rng)
        agents = scenario.setup(rng, _BOUNDS)
        assert len(agents) == scenario._num_aircraft
        assert len(scenario._obstacles) == 5


class TestMultipleResets:
    """Multiple reset/setup cycles should produce different configurations."""

    def test_different_aircraft_counts_across_resets(self) -> None:
        """Each reset produces potentially different aircraft count."""
        scenario = HorizontalCRScenario(num_aircraft=5, num_aircraft_range=(3, 8))
        counts = []
        for seed in range(20):
            rng = np.random.RandomState(seed)
            scenario.reset(rng)
            agents = scenario.setup(rng, _BOUNDS)
            counts.append(len(agents))
        # Should have at least 2 distinct counts in 20 runs
        assert len(set(counts)) >= 2

    def test_different_positions_across_resets(self) -> None:
        """Each reset produces different aircraft positions."""
        scenario = HorizontalCRScenario(num_aircraft=3, num_aircraft_range=(3, 3))
        wps_list = []
        for seed in range(5):
            rng = np.random.RandomState(seed)
            scenario.reset(rng)
            scenario.setup(rng, _BOUNDS)
            wps = {k: (v["lat"], v["lon"]) for k, v in scenario._waypoints.items()}
            wps_list.append(wps)
        # Not all waypoints should be identical
        assert not all(w == wps_list[0] for w in wps_list[1:])
