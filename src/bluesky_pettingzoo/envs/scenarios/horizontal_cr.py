"""Horizontal conflict resolution scenario.

Multiple aircraft cruise at the same altitude, use heading maneuvers
to avoid conflicts, and terminate upon reaching waypoints.
"""

from __future__ import annotations

import numpy as np

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.geometry import point_at_distance
from bluesky_pettingzoo.utils.types import AircraftState, ConflictConfig, SpawnConfig

# Constants matching bluesky-gym reference
WAYPOINT_DISTANCE_MIN_NM = 100
WAYPOINT_DISTANCE_MAX_NM = 150
CRUISE_ALT_FT = 35000.0


class HorizontalCRScenario(BaseScenario):
    """Horizontal conflict resolution scenario.

    Multiple aircraft cruise at the same altitude, use heading maneuvers
    to avoid conflicts, and terminate upon reaching waypoints.

    Args:
        num_aircraft: Number of aircraft to spawn.
        seed: Optional seed for reproducibility (used if rng not provided).
    """

    def __init__(self, num_aircraft: int = 5, seed: int | None = None) -> None:
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
        """Initialize scenario: generate agent IDs and assign waypoints.

        Aircraft are placed at random positions; waypoints are placed
        100-150 NM away in alternating east/west directions to create
        head-on conflict opportunities.
        """
        self._bounds = airspace_bounds
        self._agents = [f"AC{i:03d}" for i in range(self._num_aircraft)]
        self._waypoints = {}

        mid_lat = (airspace_bounds["lat_min"] + airspace_bounds["lat_max"]) / 2
        mid_lon = (airspace_bounds["lon_min"] + airspace_bounds["lon_max"]) / 2

        for i, acid in enumerate(self._agents):
            # Place aircraft at random position within airspace
            ac_lat = rng.uniform(airspace_bounds["lat_min"] + 0.05, airspace_bounds["lat_max"] - 0.05)
            ac_lon = rng.uniform(airspace_bounds["lon_min"] + 0.05, airspace_bounds["lon_max"] - 0.05)

            # Alternate waypoint direction: even → east, odd → west
            # This creates head-on conflict opportunities
            if i % 2 == 0:
                bearing_deg = rng.uniform(60, 120)  # roughly eastward
            else:
                bearing_deg = rng.uniform(240, 300)  # roughly westward

            dist_nm = rng.uniform(WAYPOINT_DISTANCE_MIN_NM, WAYPOINT_DISTANCE_MAX_NM)
            wp_lat, wp_lon = point_at_distance(ac_lat, ac_lon, dist_nm, bearing_deg)

            self._waypoints[acid] = {
                "lat": wp_lat,
                "lon": wp_lon,
                "alt": CRUISE_ALT_FT,
                "hdg": bearing_deg,
            }

        return list(self._agents)

    def get_spawn_config(self) -> SpawnConfig:
        """All aircraft cruise at the same altitude (horizontal CR)."""
        return SpawnConfig(
            altitude_range=(CRUISE_ALT_FT, CRUISE_ALT_FT),
            speed_range=(400, 500),
            heading_range=(0, 360),
        )

    def get_conflict_config(self) -> ConflictConfig:
        """Standard conflict thresholds for horizontal CR."""
        return ConflictConfig(
            nmac_horizontal_nm=5.0,
            nmac_vertical_ft=1000.0,
            warning_horizontal_nm=10.0,
            warning_vertical_ft=2000.0,
        )

    def get_waypoint(self, agent_id: str) -> dict[str, float]:
        """Return the assigned waypoint for an agent."""
        return self._waypoints[agent_id]
