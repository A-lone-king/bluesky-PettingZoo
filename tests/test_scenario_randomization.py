"""Tests for scenario initial position randomization (scenario-002)."""

from __future__ import annotations

import numpy as np

from bluesky_pettingzoo.envs.scenarios.horizontal_cr import HorizontalCRScenario
from bluesky_pettingzoo.envs.scenarios.vertical_cr import VerticalCRScenario


class TestHorizontalCRRandomization:
    """Test HorizontalCR scenario randomization."""

    def test_same_seed_reproducible(self) -> None:
        """Same seed should produce reproducible positions."""
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        seed = 42

        all_positions = []
        for _ in range(3):
            scenario = HorizontalCRScenario(num_aircraft=3, seed=seed)
            rng = np.random.RandomState(seed)
            scenario.setup(rng, bounds)
            positions = []
            for acid in scenario._agents:
                wp = scenario.get_waypoint(acid)
                positions.append((wp["lat"], wp["lon"]))
            all_positions.append(positions)

        # All runs with same seed should be identical
        assert all_positions[0] == all_positions[1] == all_positions[2]

    def test_different_runs_differ(self) -> None:
        """Different rng instances should produce different positions."""
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}

        positions_1 = []
        positions_2 = []

        scenario1 = HorizontalCRScenario(num_aircraft=3)
        rng1 = np.random.RandomState(42)
        scenario1.setup(rng1, bounds)
        for acid in scenario1._agents:
            wp = scenario1.get_waypoint(acid)
            positions_1.append((wp["lat"], wp["lon"]))

        scenario2 = HorizontalCRScenario(num_aircraft=3)
        rng2 = np.random.RandomState(123)
        scenario2.setup(rng2, bounds)
        for acid in scenario2._agents:
            wp = scenario2.get_waypoint(acid)
            positions_2.append((wp["lat"], wp["lon"]))

        # Different seeds should produce different positions
        assert positions_1 != positions_2


class TestVerticalCRRandomization:
    """Test VerticalCR scenario randomization."""

    def test_altitudes_shuffled(self) -> None:
        """Altitudes should be shuffled across runs."""
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}

        altitudes_list = []
        for seed in [42, 123, 456]:
            scenario = VerticalCRScenario(num_aircraft=5, seed=seed)
            rng = np.random.RandomState(seed)
            scenario.setup(rng, bounds)
            altitudes = []
            for acid in scenario._agents:
                wp = scenario.get_waypoint(acid)
                altitudes.append(wp["alt"])
            altitudes_list.append(tuple(altitudes))

        # Different seeds should produce different altitude orderings
        assert len(set(altitudes_list)) > 1

    def test_positions_randomized(self) -> None:
        """Positions should be randomized across runs."""
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}

        positions_list = []
        for seed in [42, 123, 456]:
            scenario = VerticalCRScenario(num_aircraft=3, seed=seed)
            rng = np.random.RandomState(seed)
            scenario.setup(rng, bounds)
            positions = []
            for acid in scenario._agents:
                wp = scenario.get_waypoint(acid)
                positions.append((wp["lat"], wp["lon"]))
            positions_list.append(tuple(positions))

        # Different seeds should produce different positions
        assert len(set(positions_list)) > 1


class TestProceduralGeneration:
    """Test procedural generation with num_aircraft_range."""

    def test_random_aircraft_count(self) -> None:
        """num_aircraft_range should produce different aircraft counts."""
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}

        counts = set()
        for seed in range(10):
            scenario = HorizontalCRScenario(num_aircraft_range=(3, 7), seed=seed)
            rng = np.random.RandomState(seed)
            scenario.reset(rng)
            scenario.setup(rng, bounds)
            counts.add(len(scenario._agents))

        # Should have at least 2 different counts
        assert len(counts) >= 2
