"""Extended metrics for paper-level evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExtendedMetrics:
    """Extended evaluation metrics for paper-level comparison.

    Includes safety metrics, efficiency metrics, and time metrics beyond basic reward.
    """

    conflict_resolution_rate: float
    separation_violation_duration: float
    min_separation_distance_nm: float
    trajectory_efficiency: float
    fuel_consumption_estimate: float
    mean_time_to_resolve: float


class MetricsCalculator:
    """Calculates extended metrics from episode data."""

    @staticmethod
    def calculate(
        trajectories: List[dict],
        conflicts: List[dict],
        goal_positions: dict,
        initial_distances: dict,
    ) -> ExtendedMetrics:
        """Calculate all extended metrics from episode data.

        Args:
            trajectories: List of trajectory records for each agent
            conflicts: List of conflict records during the episode
            goal_positions: Dict mapping agent_id to (lat, lon, alt) goal
            initial_distances: Dict mapping agent_id to initial distance to goal (NM)

        Returns:
            ExtendedMetrics with all calculated values
        """
        conflict_resolution_rate = MetricsCalculator._conflict_resolution_rate(conflicts)
        separation_violation_duration = MetricsCalculator._separation_violation_duration(conflicts)
        min_separation_distance = MetricsCalculator._min_separation_distance(conflicts)
        trajectory_efficiency = MetricsCalculator._trajectory_efficiency(
            trajectories, goal_positions, initial_distances
        )
        fuel_consumption = MetricsCalculator._fuel_consumption_estimate(trajectories)
        mean_time_to_resolve = MetricsCalculator._mean_time_to_resolve(conflicts)

        return ExtendedMetrics(
            conflict_resolution_rate=conflict_resolution_rate,
            separation_violation_duration=separation_violation_duration,
            min_separation_distance_nm=min_separation_distance,
            trajectory_efficiency=trajectory_efficiency,
            fuel_consumption_estimate=fuel_consumption,
            mean_time_to_resolve=mean_time_to_resolve,
        )

    @staticmethod
    def _conflict_resolution_rate(conflicts: List[dict]) -> float:
        """Calculate conflict resolution rate.

        Percentage of conflicts that were resolved (aircraft separated again after conflict).

        Args:
            conflicts: List of conflict records

        Returns:
            Resolution rate between 0.0 and 1.0
        """
        if not conflicts:
            return 1.0

        resolved = sum(1 for c in conflicts if c.get("resolved", False))
        return resolved / len(conflicts)

    @staticmethod
    def _separation_violation_duration(conflicts: List[dict]) -> float:
        """Calculate total separation violation duration across all conflicts.

        Args:
            conflicts: List of conflict records

        Returns:
            Total duration in simulation steps
        """
        total_duration = 0.0
        for c in conflicts:
            start_step = c.get("start_step", 0)
            end_step = c.get("end_step", 0)
            if end_step > start_step:
                total_duration += end_step - start_step
        return total_duration

    @staticmethod
    def _min_separation_distance(conflicts: List[dict]) -> float:
        """Calculate minimum separation distance during conflicts.

        Args:
            conflicts: List of conflict records

        Returns:
            Minimum distance in nautical miles, or infinity if no conflicts
        """
        if not conflicts:
            return float("inf")

        min_dist = float("inf")
        for c in conflicts:
            dist = c.get("min_distance_nm", float("inf"))
            if dist < min_dist:
                min_dist = dist
        return min_dist

    @staticmethod
    def _trajectory_efficiency(
        trajectories: List[dict],
        goal_positions: dict,
        initial_distances: dict,
    ) -> float:
        """Calculate trajectory efficiency.

        Efficiency = (initial_distance - final_distance) / actual_distance_traveled

        Args:
            trajectories: List of trajectory records
            goal_positions: Dict mapping agent_id to (lat, lon, alt) goal
            initial_distances: Dict mapping agent_id to initial distance to goal (NM)

        Returns:
            Mean efficiency across all agents
        """
        if not trajectories:
            return 0.0

        efficiencies = []
        for traj in trajectories:
            agent_id = traj.get("agent_id")
            if agent_id not in goal_positions or agent_id not in initial_distances:
                continue

            initial_dist = initial_distances[agent_id]
            if initial_dist <= 0:
                continue

            waypoints = traj.get("waypoints", [])
            if not waypoints:
                continue

            final_pos = waypoints[-1]
            final_lat, final_lon = final_pos.get("lat"), final_pos.get("lon")
            goal_lat, goal_lon = goal_positions[agent_id][0], goal_positions[agent_id][1]

            final_dist = MetricsCalculator._haversine_distance(final_lat, final_lon, goal_lat, goal_lon)
            progress = max(0, initial_dist - final_dist)

            actual_distance = 0.0
            for i in range(1, len(waypoints)):
                prev = waypoints[i - 1]
                curr = waypoints[i]
                actual_distance += MetricsCalculator._haversine_distance(
                    prev["lat"], prev["lon"], curr["lat"], curr["lon"]
                )

            if actual_distance > 0:
                efficiencies.append(progress / actual_distance)

        return float(sum(efficiencies) / len(efficiencies)) if efficiencies else 0.0

    @staticmethod
    def _fuel_consumption_estimate(trajectories: List[dict]) -> float:
        """Estimate fuel consumption based on trajectory data.

        Simplified model: fuel = k * distance * speed^2

        Args:
            trajectories: List of trajectory records

        Returns:
            Total estimated fuel consumption (arbitrary units)
        """
        total_fuel = 0.0
        k = 1e-6

        for traj in trajectories:
            waypoints = traj.get("waypoints", [])
            for i in range(1, len(waypoints)):
                prev = waypoints[i - 1]
                curr = waypoints[i]

                distance = MetricsCalculator._haversine_distance(
                    prev["lat"], prev["lon"], curr["lat"], curr["lon"]
                )
                avg_speed = (prev.get("speed", 0) + curr.get("speed", 0)) / 2

                total_fuel += k * distance * (avg_speed ** 2)

        return total_fuel

    @staticmethod
    def _mean_time_to_resolve(conflicts: List[dict]) -> float:
        """Calculate mean time to resolve conflicts.

        Args:
            conflicts: List of conflict records

        Returns:
            Mean resolution time in simulation steps, or infinity if no resolvable conflicts
        """
        resolved_conflicts = [c for c in conflicts if c.get("resolved", False)]
        if not resolved_conflicts:
            return float("inf")

        total_time = 0.0
        for c in resolved_conflicts:
            start_step = c.get("start_step", 0)
            end_step = c.get("end_step", 0)
            total_time += end_step - start_step

        return total_time / len(resolved_conflicts)

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in nautical miles."""
        import math

        R = 3440.065
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c