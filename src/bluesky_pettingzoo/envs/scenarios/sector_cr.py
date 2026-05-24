"""Sector conflict resolution scenario.

Multiple aircraft inside a polygon sector, use heading + speed
maneuvers to avoid conflicts. Aircraft leaving the sector are truncated.
"""

from __future__ import annotations

import numpy as np

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.geometry import generate_polygon, point_in_polygon
from bluesky_pettingzoo.utils.types import AircraftState, ConflictConfig, SpawnConfig

# Constants matching bluesky-gym reference
CRUISE_ALT_FT = 35000.0
SPEED_MIN_KT = 400.0
SPEED_MAX_KT = 500.0


class SectorCRScenario(BaseScenario):
    """Sector conflict resolution scenario.

    Aircraft are generated inside a random polygon sector. They use
    heading and speed maneuvers to avoid conflicts while remaining
    inside the sector. Aircraft leaving the sector are truncated.

    Args:
        num_aircraft: Number of aircraft to spawn.
        seed: Optional seed for reproducibility.
    """

    def __init__(self, num_aircraft: int = 5, seed: int | None = None) -> None:
        self._num_aircraft = num_aircraft
        self._seed = seed
        self._agents: list[str] = []
        self._waypoints: dict[str, dict[str, float]] = {}
        self._polygon: list[tuple[float, float]] = []
        self._bounds: dict[str, float] = {}
        self._initial_positions: dict[str, tuple[float, float]] | None = None

    @property
    def action_dimensions(self) -> list[int]:
        """Return which action indices are valid (0=heading, 1=altitude, 2=speed)."""
        return [0, 2]  # heading + speed

    def get_sector_polygon(self) -> list[tuple[float, float]]:
        """Return the polygon vertices of the sector."""
        return self._polygon

    def get_initial_positions(self) -> dict[str, tuple[float, float]] | None:
        """Return initial positions inside the polygon for each agent."""
        return self._initial_positions

    def setup(
        self,
        rng: np.random.RandomState,
        airspace_bounds: dict[str, float],
    ) -> list[str]:
        """Initialize scenario: generate polygon and aircraft inside it.

        Creates a random convex polygon sector and places aircraft
        at random positions inside it.
        """
        self._bounds = airspace_bounds
        self._agents = [f"AC{i:03d}" for i in range(self._num_aircraft)]
        self._waypoints = {}

        center_lat = (airspace_bounds["lat_min"] + airspace_bounds["lat_max"]) / 2
        center_lon = (airspace_bounds["lon_min"] + airspace_bounds["lon_max"]) / 2

        # Generate polygon sector
        self._polygon = generate_polygon(rng, center_lat, center_lon)

        # Place aircraft inside the polygon using rejection sampling
        min_lat = min(v[0] for v in self._polygon)
        max_lat = max(v[0] for v in self._polygon)
        min_lon = min(v[1] for v in self._polygon)
        max_lon = max(v[1] for v in self._polygon)

        self._initial_positions = {}
        for acid in self._agents:
            # Rejection sampling: keep generating until inside polygon
            ac_lat = center_lat
            ac_lon = center_lon
            for _ in range(1000):
                ac_lat = rng.uniform(min_lat, max_lat)
                ac_lon = rng.uniform(min_lon, max_lon)
                if point_in_polygon(ac_lat, ac_lon, self._polygon):
                    break

            self._initial_positions[acid] = (ac_lat, ac_lon)

            # Generate a waypoint on the polygon perimeter
            wp_idx = rng.randint(0, len(self._polygon))
            wp_lat, wp_lon = self._polygon[wp_idx]

            self._waypoints[acid] = {
                "lat": wp_lat,
                "lon": wp_lon,
                "alt": CRUISE_ALT_FT,
                "hdg": rng.uniform(0, 360),
            }

        return list(self._agents)

    def get_spawn_config(self) -> SpawnConfig:
        """Aircraft spawn at cruise altitude with configurable speed."""
        return SpawnConfig(
            altitude_range=(CRUISE_ALT_FT, CRUISE_ALT_FT),
            speed_range=(SPEED_MIN_KT, SPEED_MAX_KT),
            heading_range=(0, 360),
        )

    def get_conflict_config(self) -> ConflictConfig:
        """Standard conflict thresholds for sector CR."""
        return ConflictConfig(
            nmac_horizontal_nm=5.0,
            nmac_vertical_ft=1000.0,
            warning_horizontal_nm=10.0,
            warning_vertical_ft=2000.0,
        )

    def should_truncate(
        self,
        agent_id: str,
        state: AircraftState,
        airspace_bounds: dict[str, float],
    ) -> bool:
        """Truncate aircraft that leave the polygon sector."""
        return not point_in_polygon(state.lat, state.lon, self._polygon)

    def get_waypoint(self, agent_id: str) -> dict[str, float]:
        """Return the assigned waypoint for an agent."""
        return self._waypoints[agent_id]

    def get_priority(self, agent_id: str, state: AircraftState) -> float:
        """Closer to waypoint → higher priority (normalized to [-1, 1])."""
        from bluesky_pettingzoo.utils.geometry import haversine_distance

        wp = self._waypoints.get(agent_id)
        if wp is None:
            return 0.0
        dist = haversine_distance(state.lat, state.lon, wp["lat"], wp["lon"])
        # Normalize: 0 NM → priority 1.0, 50+ NM → priority -1.0
        return max(-1.0, min(1.0, 1.0 - dist / 25.0))
