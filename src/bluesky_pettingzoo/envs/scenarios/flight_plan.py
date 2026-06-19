"""Flight plan import scenario.

Loads flight plans from CSV/JSON files and creates aircraft following
their assigned routes. This scenario tests:
- Real-world flight plan execution
- Multiple aircraft with different routes and timings
- Dynamic entry based on flight plan entry times

The scenario supports flexible flight plan formats with configurable
waypoints, altitudes, and speeds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.flight_plan_parser import (
    FlightPlanData,
    FlightPlanParser,
    WaypointData,
)
from bluesky_pettingzoo.utils.geometry import haversine_distance
from bluesky_pettingzoo.utils.types import ConflictConfig, SpawnConfig


class FlightPlanScenario(BaseScenario):
    """Flight plan import scenario.

    Loads flight plans from a data file and creates aircraft that follow
    their assigned routes using LNAV. Aircraft can enter the simulation
    at different times based on their entry_time field.

    Args:
        flight_plan_path: Path to flight plan CSV or JSON file.
        num_aircraft: Maximum number of aircraft (None = use all from file).
        seed: Optional seed for reproducibility.
    """

    def __init__(
        self,
        flight_plan_path: str | None = None,
        num_aircraft: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._flight_plan_path = flight_plan_path
        self._max_aircraft = num_aircraft
        self._seed = seed
        self._agents: list[str] = []
        self._flight_plans: dict[str, FlightPlanData] = {}
        self._waypoint_indices: dict[str, int] = {}
        self._entry_times: dict[str, float] = {}
        self._bounds: dict[str, float] = {}

    @property
    def control_mode(self) -> str:
        """SINGLE_RL: ego aircraft controlled, others follow flight plans."""
        return "SINGLE_RL"

    @property
    def ego_agent(self) -> str | None:
        """Return ego agent (first aircraft)."""
        return self._agents[0] if self._agents else None

    @property
    def background_agents(self) -> list[str]:
        """Return background agents (remaining aircraft)."""
        return self._agents[1:]

    def reset(self, rng: np.random.RandomState) -> None:
        """Reset scenario state."""
        self._agents = []
        self._flight_plans = {}
        self._waypoint_indices = {}
        self._entry_times = {}

    def setup(
        self,
        rng: np.random.RandomState,
        airspace_bounds: dict[str, float],
    ) -> list[str]:
        """Initialize flight plan scenario.

        Loads flight plans and creates aircraft for each plan.

        Args:
            rng: Seeded random number generator.
            airspace_bounds: Dict with lat_min, lat_max, lon_min, lon_max.

        Returns:
            List of agent IDs.
        """
        self._bounds = airspace_bounds

        # Load flight plans
        if self._flight_plan_path:
            plans = FlightPlanParser.parse(Path(self._flight_plan_path))
            errors = FlightPlanParser.validate(plans)
            if errors:
                raise ValueError(f"Invalid flight plans: {errors}")
        else:
            # Use default flight plans for testing
            plans = self._create_default_plans()

        # Limit number of aircraft
        if self._max_aircraft is not None:
            plans = plans[: self._max_aircraft]

        # Create agents from flight plans
        for i, plan in enumerate(plans):
            agent_id = f"AC{i:03d}"
            self._agents.append(agent_id)
            self._flight_plans[agent_id] = plan
            self._waypoint_indices[agent_id] = 0
            self._entry_times[agent_id] = plan.entry_time

        return list(self._agents)

    def get_spawn_config(self) -> SpawnConfig:
        """Return spawn parameters based on first flight plan."""
        if self._agents:
            plan = self._flight_plans[self._agents[0]]
            return SpawnConfig(
                altitude_range=(plan.cruise_alt, plan.cruise_alt),
                speed_range=(plan.cruise_speed, plan.cruise_speed),
                heading_range=(0, 360),
            )
        return SpawnConfig(
            altitude_range=(35000, 35000),
            speed_range=(450, 450),
            heading_range=(0, 360),
        )

    def get_conflict_config(self) -> ConflictConfig:
        """Standard conflict thresholds."""
        return ConflictConfig(
            nmac_horizontal_nm=5.0,
            nmac_vertical_ft=1000.0,
            warning_horizontal_nm=10.0,
            warning_vertical_ft=2000.0,
        )

    def get_initial_positions(self) -> dict[str, tuple[float, float]] | None:
        """Place aircraft at their first waypoint or origin."""
        positions = {}
        for agent_id, plan in self._flight_plans.items():
            if plan.waypoints:
                # Start at first waypoint
                wp = plan.waypoints[0]
                positions[agent_id] = (wp.lat, wp.lon)
            else:
                # Default position near center of bounds
                center_lat = (self._bounds["lat_min"] + self._bounds["lat_max"]) / 2
                center_lon = (self._bounds["lon_min"] + self._bounds["lon_max"]) / 2
                positions[agent_id] = (center_lat, center_lon)
        return positions

    def get_waypoint(self, agent_id: str) -> dict[str, float]:
        """Return the current waypoint for an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            Dict with lat, lon, alt, hdg keys.
        """
        plan = self._flight_plans.get(agent_id)
        if plan is None:
            return {"lat": 0.0, "lon": 0.0, "alt": 35000.0, "hdg": 0.0}

        idx = self._waypoint_indices.get(agent_id, 0)
        if idx >= len(plan.waypoints):
            # Use destination if no more waypoints
            return {
                "lat": self._bounds.get("lat_min", 52.0),
                "lon": self._bounds.get("lon_min", 4.0),
                "alt": plan.cruise_alt,
                "hdg": 0.0,
            }

        wp = plan.waypoints[idx]

        # Calculate heading to next waypoint
        hdg = 0.0
        if idx + 1 < len(plan.waypoints):
            next_wp = plan.waypoints[idx + 1]
            hdg = self._calculate_heading(wp.lat, wp.lon, next_wp.lat, next_wp.lon)

        return {
            "lat": wp.lat,
            "lon": wp.lon,
            "alt": wp.alt,
            "hdg": hdg,
        }

    def update_waypoint(
        self,
        agent_id: str,
        state: Any,
    ) -> dict[str, float] | None:
        """Advance to next waypoint when current one is reached.

        Args:
            agent_id: Agent identifier.
            state: Current aircraft state.

        Returns:
            Next waypoint dict, or None if route complete.
        """
        plan = self._flight_plans.get(agent_id)
        if plan is None:
            return None

        idx = self._waypoint_indices.get(agent_id, 0)
        if idx + 1 >= len(plan.waypoints):
            # Route complete
            return None

        # Check if close enough to current waypoint (< 2 NM)
        current_wp = plan.waypoints[idx]
        dist = haversine_distance(state.lat, state.lon, current_wp.lat, current_wp.lon)
        if dist < 2.0:
            self._waypoint_indices[agent_id] = idx + 1
            next_wp = plan.waypoints[idx + 1]
            return {
                "lat": next_wp.lat,
                "lon": next_wp.lon,
            }

        return None

    def configure_npc_navigation(self, wrapper: Any) -> list[str]:
        """Configure LNAV for all aircraft to follow their flight plans.

        Args:
            wrapper: BlueSkyWrapper instance.

        Returns:
            List of BlueSky commands sent.
        """
        commands: list[str] = []

        for agent_id, plan in self._flight_plans.items():
            if not plan.waypoints:
                continue

            # Set origin at first waypoint
            first_wp = plan.waypoints[0]
            wrapper.set_origin(agent_id, first_wp.lat, first_wp.lon)
            commands.append(f"ORIG {agent_id} {first_wp.lat} {first_wp.lon}")

            # Set destination at last waypoint
            last_wp = plan.waypoints[-1]
            wrapper.set_destination(agent_id, last_wp.lat, last_wp.lon)
            commands.append(f"DEST {agent_id} {last_wp.lat} {last_wp.lon}")

            # Add intermediate waypoints (skip first and last)
            for wp in plan.waypoints[1:-1]:
                wrapper.add_waypoint(agent_id, wp.lat, wp.lon)
                commands.append(f"ADDWPT {agent_id} {wp.lat} {wp.lon}")

            # Enable LNAV
            wrapper.enable_lnav(agent_id)
            commands.append(f"LNAV {agent_id} ON")

            # Set initial speed
            wrapper.send_command(f"SPD {agent_id} {plan.cruise_speed}")
            commands.append(f"SPD {agent_id} {plan.cruise_speed}")

        return commands

    def update(
        self,
        step_count: int,
        all_states: dict[str, Any],
    ) -> list[str]:
        """Called each step; handle dynamic entry based on entry_time.

        Args:
            step_count: Current step number.
            all_states: All current aircraft states.

        Returns:
            List of new agent ID strings (empty by default).
        """
        # Dynamic entry is handled by the environment based on entry_times
        return []

    def get_entry_times(self) -> dict[str, float]:
        """Return entry times for each agent.

        Returns:
            Dict mapping agent ID to entry time in seconds.
        """
        return dict(self._entry_times)

    def _create_default_plans(self) -> list[FlightPlanData]:
        """Create default flight plans for testing."""

        return [
            FlightPlanData(
                flight_id="FL001",
                aircraft_type="B738",
                origin="EHAM",
                destination="EBBR",
                entry_time=0.0,
                cruise_alt=35000.0,
                cruise_speed=450.0,
                waypoints=[
                    WaypointData("ARTIP", 52.8, 5.5, 35000.0),
                    WaypointData("SULEG", 52.6, 5.0, 35000.0),
                    WaypointData("AUDIK", 52.4, 4.8, 35000.0),
                    WaypointData("EBBR", 50.9, 4.5, 0.0),
                ],
            ),
            FlightPlanData(
                flight_id="FL002",
                aircraft_type="A320",
                origin="EGLL",
                destination="EHAM",
                entry_time=30.0,
                cruise_alt=33000.0,
                cruise_speed=430.0,
                waypoints=[
                    WaypointData("BASVO", 51.5, -1.0, 33000.0),
                    WaypointData("RIVER", 51.5, 4.8, 33000.0),
                    WaypointData("VEDON", 51.7, 4.6, 33000.0),
                    WaypointData("EHAM", 52.3, 4.8, 0.0),
                ],
            ),
            FlightPlanData(
                flight_id="FL003",
                aircraft_type="B77W",
                origin="LFPG",
                destination="EHAM",
                entry_time=60.0,
                cruise_alt=37000.0,
                cruise_speed=480.0,
                waypoints=[
                    WaypointData("SOBTU", 49.0, 2.5, 37000.0),
                    WaypointData("IMDIK", 50.5, 3.5, 37000.0),
                    WaypointData("NEKNU", 51.5, 4.2, 37000.0),
                    WaypointData("EHAM", 52.3, 4.8, 0.0),
                ],
            ),
        ]

    def _calculate_heading(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate initial bearing between two points.

        Args:
            lat1, lon1: Start point.
            lat2, lon2: End point.

        Returns:
            Bearing in degrees (0-360).
        """
        import math

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)

        x = math.sin(dlon_rad) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
            lat2_rad
        ) * math.cos(dlon_rad)

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360
