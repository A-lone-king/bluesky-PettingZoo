"""Unit tests for extended evaluation metrics."""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.training.metrics import ExtendedMetrics, MetricsCalculator


class TestExtendedMetrics:
    """Tests for ExtendedMetrics dataclass."""

    def test_creation(self):
        """Test ExtendedMetrics creation with all fields."""
        metrics = ExtendedMetrics(
            conflict_resolution_rate=0.85,
            separation_violation_duration=15.0,
            min_separation_distance_nm=1.5,
            trajectory_efficiency=0.72,
            fuel_consumption_estimate=125.5,
            mean_time_to_resolve=8.0,
        )
        assert metrics.conflict_resolution_rate == 0.85
        assert metrics.separation_violation_duration == 15.0
        assert metrics.min_separation_distance_nm == 1.5
        assert metrics.trajectory_efficiency == 0.72
        assert metrics.fuel_consumption_estimate == 125.5
        assert metrics.mean_time_to_resolve == 8.0


class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""

    def test_conflict_resolution_rate_all_resolved(self):
        """Test conflict resolution rate with all conflicts resolved."""
        conflicts = [
            {"resolved": True, "start_step": 10, "end_step": 20},
            {"resolved": True, "start_step": 30, "end_step": 35},
        ]
        rate = MetricsCalculator._conflict_resolution_rate(conflicts)
        assert rate == 1.0

    def test_conflict_resolution_rate_partial(self):
        """Test conflict resolution rate with partial resolution."""
        conflicts = [
            {"resolved": True, "start_step": 10, "end_step": 20},
            {"resolved": False, "start_step": 30, "end_step": 40},
            {"resolved": True, "start_step": 50, "end_step": 55},
        ]
        rate = MetricsCalculator._conflict_resolution_rate(conflicts)
        assert abs(rate - 2/3) < 0.001

    def test_conflict_resolution_rate_no_conflicts(self):
        """Test conflict resolution rate with no conflicts."""
        rate = MetricsCalculator._conflict_resolution_rate([])
        assert rate == 1.0

    def test_separation_violation_duration(self):
        """Test separation violation duration calculation."""
        conflicts = [
            {"start_step": 10, "end_step": 20},
            {"start_step": 30, "end_step": 35},
            {"start_step": 50, "end_step": 45},
        ]
        duration = MetricsCalculator._separation_violation_duration(conflicts)
        assert duration == 15.0

    def test_min_separation_distance(self):
        """Test minimum separation distance calculation."""
        conflicts = [
            {"min_distance_nm": 2.0},
            {"min_distance_nm": 1.5},
            {"min_distance_nm": 3.0},
        ]
        min_dist = MetricsCalculator._min_separation_distance(conflicts)
        assert min_dist == 1.5

    def test_min_separation_distance_no_conflicts(self):
        """Test minimum separation distance with no conflicts."""
        min_dist = MetricsCalculator._min_separation_distance([])
        assert min_dist == float("inf")

    def test_trajectory_efficiency(self):
        """Test trajectory efficiency calculation."""
        trajectories = [
            {
                "agent_id": "AC001",
                "waypoints": [
                    {"lat": 50.0, "lon": 0.0, "speed": 450},
                    {"lat": 50.1, "lon": 0.1, "speed": 450},
                    {"lat": 50.2, "lon": 0.2, "speed": 450},
                ],
            }
        ]
        goal_positions = {"AC001": (50.2, 0.2, 30000)}
        initial_distances = {"AC001": 20.0}

        efficiency = MetricsCalculator._trajectory_efficiency(
            trajectories, goal_positions, initial_distances
        )
        assert efficiency >= 0.0
        assert efficiency <= 1.0

    def test_trajectory_efficiency_no_trajectories(self):
        """Test trajectory efficiency with no trajectories."""
        efficiency = MetricsCalculator._trajectory_efficiency([], {}, {})
        assert efficiency == 0.0

    def test_fuel_consumption_estimate(self):
        """Test fuel consumption estimation."""
        trajectories = [
            {
                "agent_id": "AC001",
                "waypoints": [
                    {"lat": 50.0, "lon": 0.0, "speed": 450},
                    {"lat": 50.1, "lon": 0.1, "speed": 450},
                ],
            }
        ]
        fuel = MetricsCalculator._fuel_consumption_estimate(trajectories)
        assert fuel >= 0.0

    def test_fuel_consumption_no_trajectories(self):
        """Test fuel consumption with no trajectories."""
        fuel = MetricsCalculator._fuel_consumption_estimate([])
        assert fuel == 0.0

    def test_mean_time_to_resolve(self):
        """Test mean time to resolve calculation."""
        conflicts = [
            {"resolved": True, "start_step": 10, "end_step": 20},
            {"resolved": True, "start_step": 30, "end_step": 35},
            {"resolved": False, "start_step": 40, "end_step": 50},
        ]
        time = MetricsCalculator._mean_time_to_resolve(conflicts)
        assert time == 7.5

    def test_mean_time_to_resolve_no_resolved(self):
        """Test mean time to resolve with no resolved conflicts."""
        conflicts = [
            {"resolved": False, "start_step": 10, "end_step": 20},
        ]
        time = MetricsCalculator._mean_time_to_resolve(conflicts)
        assert time == float("inf")

    def test_mean_time_to_resolve_no_conflicts(self):
        """Test mean time to resolve with no conflicts."""
        time = MetricsCalculator._mean_time_to_resolve([])
        assert time == float("inf")

    def test_haversine_distance(self):
        """Test haversine distance calculation."""
        dist = MetricsCalculator._haversine_distance(50.0, 0.0, 50.0, 1.0)
        assert dist > 0.0
        assert dist < 100.0

    def test_calculate_full(self):
        """Test full metrics calculation."""
        trajectories = [
            {
                "agent_id": "AC001",
                "waypoints": [
                    {"lat": 50.0, "lon": 0.0, "speed": 450},
                    {"lat": 50.1, "lon": 0.1, "speed": 450},
                ],
            }
        ]
        conflicts = [
            {"resolved": True, "start_step": 10, "end_step": 20, "min_distance_nm": 1.5},
        ]
        goal_positions = {"AC001": (50.2, 0.2, 30000)}
        initial_distances = {"AC001": 20.0}

        metrics = MetricsCalculator.calculate(
            trajectories, conflicts, goal_positions, initial_distances
        )

        assert isinstance(metrics, ExtendedMetrics)
        assert metrics.conflict_resolution_rate == 1.0
        assert metrics.separation_violation_duration == 10.0
        assert metrics.min_separation_distance_nm == 1.5
        assert metrics.trajectory_efficiency >= 0.0
        assert metrics.fuel_consumption_estimate >= 0.0
        assert metrics.mean_time_to_resolve == 10.0