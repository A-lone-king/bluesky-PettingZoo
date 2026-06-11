"""Tests for scenario complexity enhancements (scenario-enhance-001).

Enhances three core scenarios:
- HorizontalCR: Multi-altitude layer conflicts (3-4 layers, 2-3 aircraft per layer)
- VerticalCR: Real approach profile (3° glide slope, speed constraints)
- SectorCR: Dynamic capacity changes (simulating peak/off-peak periods)
"""

from __future__ import annotations

import numpy as np

from bluesky_pettingzoo.envs.scenarios.horizontal_cr import HorizontalCRScenario
from bluesky_pettingzoo.envs.scenarios.sector_cr import SectorCRScenario
from bluesky_pettingzoo.envs.scenarios.vertical_cr import VerticalCRScenario
from bluesky_pettingzoo.utils.types import AircraftState

# ===========================================================================
# HorizontalCR: Multi-altitude layer conflict tests
# ===========================================================================


class TestHorizontalCRMultiAltitude:
    """Test HorizontalCR multi-altitude layer enhancement."""

    def test_setup_creates_multiple_altitude_layers(self) -> None:
        """Aircraft should be distributed across 3-4 altitude layers."""
        scenario = HorizontalCRScenario(num_aircraft=8, num_altitude_layers=3)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        agents = scenario.setup(rng, bounds)

        assert len(agents) == 8
        # Check altitude layers exist
        altitudes = set()
        for wp in scenario._waypoints.values():
            altitudes.add(wp["alt"])
        assert len(altitudes) == 3, f"Expected 3 altitude layers, got {len(altitudes)}"

    def test_aircraft_per_layer_within_range(self) -> None:
        """Each altitude layer should have 2-3 aircraft."""
        scenario = HorizontalCRScenario(num_aircraft=9, num_altitude_layers=3)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)

        # Count aircraft per layer
        layer_counts: dict[float, int] = {}
        for wp in scenario._waypoints.values():
            alt = wp["alt"]
            layer_counts[alt] = layer_counts.get(alt, 0) + 1

        for alt, count in layer_counts.items():
            assert 2 <= count <= 3, f"Layer at {alt} ft has {count} aircraft (expected 2-3)"

    def test_default_layers_is_one(self) -> None:
        """Default behavior should have 1 altitude layer (backward compatible)."""
        scenario = HorizontalCRScenario(num_aircraft=5)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)

        altitudes = set()
        for wp in scenario._waypoints.values():
            altitudes.add(wp["alt"])
        assert len(altitudes) == 1, "Default should have 1 altitude layer"

    def test_procedural_generation_with_layers(self) -> None:
        """Procedural generation should work with multi-altitude layers."""
        scenario = HorizontalCRScenario(
            num_aircraft=6,
            num_aircraft_range=(4, 8),
            num_altitude_layers=2,
        )
        rng = np.random.RandomState(42)
        scenario.reset(rng)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        agents = scenario.setup(rng, bounds)

        assert 4 <= len(agents) <= 8
        altitudes = set()
        for wp in scenario._waypoints.values():
            altitudes.add(wp["alt"])
        assert len(altitudes) == 2

    def test_altitude_separation_sufficient(self) -> None:
        """Altitude layers should be separated by at least 2000 ft."""
        scenario = HorizontalCRScenario(num_aircraft=8, num_altitude_layers=3)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)

        alts = sorted(set(wp["alt"] for wp in scenario._waypoints.values()))
        for i in range(len(alts) - 1):
            separation = alts[i + 1] - alts[i]
            assert separation >= 2000, (
                f"Altitude separation {separation} ft < 2000 ft between layers"
            )


# ===========================================================================
# VerticalCR: Real approach profile tests
# ===========================================================================


class TestVerticalCRApproachProfile:
    """Test VerticalCR real approach profile enhancement."""

    def test_setup_creates_approach_profile(self) -> None:
        """Aircraft should have approach profiles with glide slope."""
        scenario = VerticalCRScenario(
            num_aircraft=3,
            use_approach_profile=True,
        )
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        agents = scenario.setup(rng, bounds)

        assert len(agents) == 3
        assert hasattr(scenario, "_approach_profiles")
        assert len(scenario._approach_profiles) == 3

    def test_approach_profile_has_glide_slope(self) -> None:
        """Approach profile should have 3° glide slope."""
        scenario = VerticalCRScenario(
            num_aircraft=3,
            use_approach_profile=True,
        )
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)

        for acid, profile in scenario._approach_profiles.items():
            assert "glide_slope_deg" in profile
            assert profile["glide_slope_deg"] == 3.0

    def test_approach_profile_has_speed_constraints(self) -> None:
        """Approach profile should have speed constraints."""
        scenario = VerticalCRScenario(
            num_aircraft=3,
            use_approach_profile=True,
        )
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)

        for acid, profile in scenario._approach_profiles.items():
            assert "initial_speed_kt" in profile
            assert "final_speed_kt" in profile
            assert "final_alt_ft" in profile
            assert profile["initial_speed_kt"] > profile["final_speed_kt"]

    def test_default_no_approach_profile(self) -> None:
        """Default behavior should not use approach profile (backward compatible)."""
        scenario = VerticalCRScenario(num_aircraft=5)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)

        assert not hasattr(scenario, "_approach_profiles") or len(scenario._approach_profiles) == 0

    def test_approach_profile_termination_condition(self) -> None:
        """Aircraft should terminate when reaching final altitude."""
        scenario = VerticalCRScenario(
            num_aircraft=3,
            use_approach_profile=True,
        )
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)

        for acid, profile in scenario._approach_profiles.items():
            # Create a state near final altitude
            state = AircraftState(
                id=acid,
                lat=40.0,
                lon=117.0,
                alt=profile["final_alt_ft"] + 100,  # Near final altitude
                hdg=180.0,
                vs=profile["target_vs_ft_min"],
                tas=profile["final_speed_kt"],
            )
            # Should truncate when within 500 ft of final altitude
            assert scenario.should_truncate(acid, state, bounds) is True


# ===========================================================================
# SectorCR: Dynamic capacity tests
# ===========================================================================


class TestSectorCRDynamicCapacity:
    """Test SectorCR dynamic capacity enhancement."""

    def test_setup_with_dynamic_capacity(self) -> None:
        """Sector should support dynamic capacity changes."""
        scenario = SectorCRScenario(
            num_aircraft=5,
            use_dynamic_capacity=True,
        )
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        agents = scenario.setup(rng, bounds)

        assert len(agents) == 5
        assert hasattr(scenario, "_capacity_schedule")
        assert len(scenario._capacity_schedule) > 0

    def test_capacity_schedule_has_peak_offpeak(self) -> None:
        """Capacity schedule should have peak and off-peak periods."""
        scenario = SectorCRScenario(
            num_aircraft=5,
            use_dynamic_capacity=True,
        )
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)

        capacities = [entry["capacity"] for entry in scenario._capacity_schedule]
        assert min(capacities) < max(capacities), "Should have varying capacity levels"

    def test_capacity_changes_over_time(self) -> None:
        """Capacity should change based on step count."""
        scenario = SectorCRScenario(
            num_aircraft=5,
            use_dynamic_capacity=True,
        )
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)

        # Get capacity at different steps
        cap_step_0 = scenario.get_current_capacity(0)
        cap_step_50 = scenario.get_current_capacity(50)
        cap_step_100 = scenario.get_current_capacity(100)

        # At least one should differ
        assert not (cap_step_0 == cap_step_50 == cap_step_100), "Capacity should change over time"

    def test_truncation_respects_dynamic_capacity(self) -> None:
        """Aircraft should be truncated when capacity exceeded."""
        scenario = SectorCRScenario(
            num_aircraft=3,
            use_dynamic_capacity=True,
            capacity_min=2,
            capacity_max=2,  # Fixed capacity of 2
        )
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)

        # Create states for all aircraft inside polygon
        states = {}
        for acid in scenario._agents:
            states[acid] = AircraftState(
                id=acid,
                lat=40.0,
                lon=117.0,
                alt=35000.0,
                hdg=90.0,
                vs=0.0,
                tas=450.0,
            )

        # First 2 should not truncate, 3rd should (capacity = 2)
        truncation_results = []
        for acid in scenario._agents:
            result = scenario.should_truncate(acid, states[acid], bounds)
            truncation_results.append(result)

        # At least one should be truncated due to capacity
        assert any(truncation_results), "At least one aircraft should be truncated due to capacity"

    def test_default_no_dynamic_capacity(self) -> None:
        """Default behavior should not use dynamic capacity (backward compatible)."""
        scenario = SectorCRScenario(num_aircraft=5)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)

        assert not hasattr(scenario, "_capacity_schedule") or len(scenario._capacity_schedule) == 0

    def test_capacity_schedule_configuration(self) -> None:
        """Capacity schedule should be configurable."""
        scenario = SectorCRScenario(
            num_aircraft=5,
            use_dynamic_capacity=True,
            capacity_schedule=[
                {"start_step": 0, "end_step": 50, "capacity": 3},
                {"start_step": 50, "end_step": 100, "capacity": 5},
            ],
        )
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)

        assert scenario.get_current_capacity(0) == 3
        assert scenario.get_current_capacity(50) == 5
