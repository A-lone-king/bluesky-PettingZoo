"""Waypoint navigation scenario (no conflicts).

Aircraft navigate to assigned waypoints without conflict opportunities.
Used as a baseline for testing guidance logic and arrival termination.
"""

from __future__ import annotations

import numpy as np

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.geometry import haversine_distance, point_at_distance
from bluesky_pettingzoo.utils.types import AircraftState, ConflictConfig, SpawnConfig

# Constants
WAYPOINT_DISTANCE_MIN_NM = 80
WAYPOINT_DISTANCE_MAX_NM = 150
CRUISE_ALT_FT = 35000.0
MIN_SEPARATION_NM = 30.0  # Minimum distance between any two aircraft


class WaypointNavScenario(BaseScenario):
    """Pure navigation task without conflicts.

    Aircraft are placed far apart (minimum 30 NM separation) and each
    navigates to an assigned waypoint.  No conflict opportunities exist.

    Args:
        num_aircraft: Number of aircraft to spawn.
        seed: Optional seed for reproducibility.
    """

    def __init__(self, num_aircraft: int = 3, seed: int | None = None) -> None:
        self._num_aircraft = num_aircraft
        self._seed = seed
        self._agents: list[str] = []
        self._waypoints: dict[str, dict[str, float]] = {}
        self._bounds: dict[str, float] = {}

    @property
    def action_dimensions(self) -> list[int]:
        """Return which action indices are valid (0=heading, 1=altitude, 2=speed)."""
        return [0]  # heading only

    def setup(
        self,
        rng: np.random.RandomState,
        airspace_bounds: dict[str, float],
    ) -> list[str]:
        """Initialize scenario: generate well-separated aircraft with waypoints.

        Aircraft are placed with at least MIN_SEPARATION_NM between any pair
        to ensure no conflicts occur.
        """
        self._bounds = airspace_bounds
        self._agents = [f"AC{i:03d}" for i in range(self._num_aircraft)]
        self._waypoints = {}

        lat_min = airspace_bounds["lat_min"] + 0.1
        lat_max = airspace_bounds["lat_max"] - 0.1
        lon_min = airspace_bounds["lon_min"] + 0.1
        lon_max = airspace_bounds["lon_max"] - 0.1

        # Place aircraft using rejection sampling to ensure minimum separation
        positions: list[tuple[float, float]] = []
        for acid in self._agents:
            for _ in range(1000):
                ac_lat = rng.uniform(lat_min, lat_max)
                ac_lon = rng.uniform(lon_min, lon_max)
                # Check distance to all existing aircraft
                too_close = False
                for plat, plon in positions:
                    if haversine_distance(ac_lat, ac_lon, plat, plon) < MIN_SEPARATION_NM:
                        too_close = True
                        break
                if not too_close:
                    positions.append((ac_lat, ac_lon))
                    break

            # Assign waypoint in a random direction
            bearing = rng.uniform(0, 360)
            dist = rng.uniform(WAYPOINT_DISTANCE_MIN_NM, WAYPOINT_DISTANCE_MAX_NM)
            wp_lat, wp_lon = point_at_distance(ac_lat, ac_lon, dist, bearing)

            self._waypoints[acid] = {
                "lat": wp_lat,
                "lon": wp_lon,
                "alt": CRUISE_ALT_FT,
                "hdg": bearing,
            }

        return list(self._agents)

    def get_spawn_config(self) -> SpawnConfig:
        """All aircraft at cruise altitude."""
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
        """Return the assigned waypoint for an agent."""
        return self._waypoints[agent_id]
