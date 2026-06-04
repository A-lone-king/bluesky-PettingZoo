"""PlanWaypoint scenario — sequential waypoint navigation.

Single aircraft visits 5 waypoints in order using heading-only control.
Each waypoint arrival is detected when the aircraft is within a threshold
distance. Reaching all waypoints completes the episode.
"""

from __future__ import annotations

import numpy as np

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.geometry import haversine_distance, point_at_distance
from bluesky_pettingzoo.utils.types import AircraftState, ConflictConfig, SpawnConfig

# Constants
CRUISE_ALT_FT = 35000.0
NUM_WAYPOINTS = 5
WAYPOINT_SPACING_NM = 30.0
ARRIVAL_THRESHOLD_NM = 2.0


class PlanWaypointScenario(BaseScenario):
    """Sequential waypoint navigation scenario.

    A single aircraft visits 5 waypoints in order. Only heading control
    is used. Each waypoint arrival clears the waypoint and advances to
    the next one.

    Args:
        seed: Optional seed for reproducibility.
    """

    def __init__(
        self,
        num_aircraft: int = 1,
        num_aircraft_range: tuple[int, int] | None = None,
        seed: int | None = None,
    ) -> None:
        self._num_aircraft = num_aircraft
        self._num_aircraft_range = num_aircraft_range
        self._seed = seed
        self._agents: list[str] = []
        self._waypoints: list[dict[str, float]] = []
        self._reached: list[bool] = []
        self._bounds: dict[str, float] = {}
        self._start_lat: float = 0.0
        self._start_lon: float = 0.0

    @property
    def action_dimensions(self) -> list[int]:
        """Heading only."""
        return [0]

    @property
    def num_aircraft_range(self) -> tuple[int, int] | None:
        """Return dynamic aircraft count range if configured."""
        return self._num_aircraft_range

    def reset(self, rng: np.random.RandomState) -> None:
        """Randomize aircraft count and clear state for procedural generation."""
        if self._num_aircraft_range is not None:
            self._num_aircraft = int(rng.randint(
                self._num_aircraft_range[0],
                self._num_aircraft_range[1] + 1,
            ))
        self._agents = []
        self._waypoints = []
        self._reached = []

    def setup(
        self,
        rng: np.random.RandomState,
        airspace_bounds: dict[str, float],
    ) -> list[str]:
        """Initialize scenario: 1 agent, 5 sequential waypoints.

        Waypoints are placed in a chain starting from the aircraft's
        position, each ~30 NM apart in a roughly straight line.
        """
        self._bounds = airspace_bounds
        self._agents = ["AC000"]

        # Start position near center
        self._start_lat = (airspace_bounds["lat_min"] + airspace_bounds["lat_max"]) / 2
        self._start_lon = (airspace_bounds["lon_min"] + airspace_bounds["lon_max"]) / 2

        # Generate 5 waypoints in a chain
        self._waypoints = []
        self._reached = [False] * NUM_WAYPOINTS

        base_bearing = rng.uniform(0, 360)
        lat, lon = self._start_lat, self._start_lon
        for i in range(NUM_WAYPOINTS):
            bearing = base_bearing + rng.uniform(-15, 15)
            wp_lat, wp_lon = point_at_distance(lat, lon, WAYPOINT_SPACING_NM, bearing)
            self._waypoints.append(
                {
                    "lat": wp_lat,
                    "lon": wp_lon,
                    "alt": CRUISE_ALT_FT,
                    "hdg": bearing,
                }
            )
            lat, lon = wp_lat, wp_lon

        return list(self._agents)

    def get_spawn_config(self) -> SpawnConfig:
        """Aircraft at cruise altitude."""
        return SpawnConfig(
            altitude_range=(CRUISE_ALT_FT, CRUISE_ALT_FT),
            speed_range=(400, 500),
            heading_range=(0, 360),
        )

    def get_conflict_config(self) -> ConflictConfig:
        """Standard conflict thresholds (not expected to trigger)."""
        return ConflictConfig(
            nmac_horizontal_nm=5.0,
            nmac_vertical_ft=1000.0,
            warning_horizontal_nm=10.0,
            warning_vertical_ft=2000.0,
        )

    def get_waypoint(self, agent_id: str) -> dict[str, float]:
        """Return the current (next unreached) waypoint."""
        for i, reached in enumerate(self._reached):
            if not reached:
                return self._waypoints[i]
        # All reached — return last waypoint
        return self._waypoints[-1]

    def get_current_waypoint_index(self) -> int:
        """Return index of the current waypoint, or -1 if all reached."""
        for i, reached in enumerate(self._reached):
            if not reached:
                return i
        return -1

    def mark_reached(self, index: int) -> None:
        """Mark a waypoint as reached."""
        if 0 <= index < len(self._reached):
            self._reached[index] = True

    def get_reached_count(self) -> int:
        """Return number of waypoints reached."""
        return sum(self._reached)

    def all_reached(self) -> bool:
        """Return True if all waypoints have been reached."""
        return all(self._reached)

    def check_arrival(self, agent_id: str, state: AircraftState) -> int | None:
        """Check if aircraft has arrived at the current waypoint.

        If arrived, marks the waypoint as reached and returns its index.

        Args:
            agent_id: Agent identifier.
            state: Current aircraft state.

        Returns:
            Index of reached waypoint, or None if not arrived.
        """
        idx = self.get_current_waypoint_index()
        if idx < 0:
            return None
        wp = self._waypoints[idx]
        dist = haversine_distance(state.lat, state.lon, wp["lat"], wp["lon"])
        if dist < ARRIVAL_THRESHOLD_NM:
            self.mark_reached(idx)
            return idx
        return None

    def get_initial_positions(self) -> dict[str, tuple[float, float]] | None:
        """Place aircraft at the start position."""
        return {"AC000": (self._start_lat, self._start_lon)}

    def get_priority(self, agent_id: str, state: AircraftState) -> float:
        """Default equal priority."""
        return 0.0
