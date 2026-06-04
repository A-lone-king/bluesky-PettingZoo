"""Tests for SectorCapacityScenario."""

from __future__ import annotations

import numpy as np

from bluesky_pettingzoo.envs.scenarios.sector_capacity import SectorCapacityScenario
from bluesky_pettingzoo.utils.types import AircraftState


def _make_state(acid: str, lat: float, lon: float) -> AircraftState:
    return AircraftState(id=acid, lat=lat, lon=lon, alt=35000.0, hdg=90.0, tas=450.0, vs=0.0)


class TestSectorCapacitySetup:
    """Scenario setup should create sectors and agents."""

    def test_setup_returns_correct_agents(self) -> None:
        scenario = SectorCapacityScenario(num_aircraft=6, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 117.0}
        agents = scenario.setup(rng, bounds)
        assert len(agents) == 6
        assert agents == [f"AC{i:03d}" for i in range(6)]

    def test_setup_creates_sectors(self) -> None:
        scenario = SectorCapacityScenario(num_aircraft=4, num_sectors=3, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 117.0}
        scenario.setup(rng, bounds)
        sectors = scenario.get_sectors()
        assert len(sectors) == 3
        for s in sectors:
            assert "id" in s
            assert "bounds" in s
            assert "capacity" in s

    def test_sector_capacity_propagated(self) -> None:
        scenario = SectorCapacityScenario(num_aircraft=4, sector_capacity=5, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 117.0}
        scenario.setup(rng, bounds)
        for s in scenario.get_sectors():
            assert s["capacity"] == 5


class TestSectorCapacityPositions:
    """Aircraft should start in the first sector."""

    def test_aircraft_in_first_sector(self) -> None:
        scenario = SectorCapacityScenario(num_aircraft=4, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 117.0}
        scenario.setup(rng, bounds)
        positions = scenario.get_initial_positions()
        sectors = scenario.get_sectors()
        first_sector = sectors[0]
        (lat_min, lon_min), (lat_max, lon_max) = first_sector["bounds"]

        for acid, (lat, lon) in positions.items():
            assert lat_min <= lat <= lat_max, f"{acid} lat out of first sector"
            assert lon_min <= lon <= lon_max, f"{acid} lon out of first sector"


class TestSectorCapacityWaypoints:
    """Each agent should have a waypoint."""

    def test_all_agents_have_waypoints(self) -> None:
        scenario = SectorCapacityScenario(num_aircraft=4, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 117.0}
        agents = scenario.setup(rng, bounds)
        for acid in agents:
            wp = scenario.get_waypoint(acid)
            assert "lat" in wp
            assert "lon" in wp
            assert "alt" in wp


class TestSectorCapacityTruncation:
    """Aircraft leaving all sectors should be truncated."""

    def test_outside_sectors_truncated(self) -> None:
        scenario = SectorCapacityScenario(num_aircraft=2, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 117.0}
        scenario.setup(rng, bounds)
        state = _make_state("AC000", lat=40.0, lon=116.5)  # outside
        assert scenario.should_truncate("AC000", state, bounds) is True

    def test_inside_sectors_not_truncated(self) -> None:
        scenario = SectorCapacityScenario(num_aircraft=2, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 117.0}
        scenario.setup(rng, bounds)
        state = _make_state("AC000", lat=39.25, lon=116.25)
        assert scenario.should_truncate("AC000", state, bounds) is False


class TestSectorCapacityActionDimensions:
    """Action dimensions should be heading + speed."""

    def test_action_dimensions(self) -> None:
        scenario = SectorCapacityScenario()
        assert scenario.action_dimensions == [0, 2]


class TestSectorCapacitySpawnConfig:
    """Spawn config should use cruise altitude and 400-500kt."""

    def test_spawn_config(self) -> None:
        scenario = SectorCapacityScenario()
        cfg = scenario.get_spawn_config()
        assert cfg.altitude_range == (35000.0, 35000.0)
        assert cfg.speed_range == (400.0, 500.0)
