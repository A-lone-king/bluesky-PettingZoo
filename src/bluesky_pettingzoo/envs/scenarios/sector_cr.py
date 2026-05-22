"""Sector conflict resolution scenario.

Multiple aircraft inside a polygon sector, use heading + speed
maneuvers to avoid conflicts. Aircraft leaving the sector are truncated.
"""

from __future__ import annotations

import math

import numpy as np

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.types import AircraftState, ConflictConfig, SpawnConfig

# Constants matching bluesky-gym reference
CRUISE_ALT_FT = 35000.0
SPEED_MIN_KT = 400.0
SPEED_MAX_KT = 500.0


def _point_in_polygon(lat: float, lon: float, polygon: list[tuple[float, float]]) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm.

    Args:
        lat: Point latitude.
        lon: Point longitude.
        polygon: List of (lat, lon) vertices.

    Returns:
        True if point is inside the polygon.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _generate_polygon(
    rng: np.random.RandomState,
    center_lat: float,
    center_lon: float,
    num_vertices: int = 6,
    radius_deg: float = 0.15,
) -> list[tuple[float, float]]:
    """Generate a random convex polygon around a center point.

    Args:
        rng: Random number generator.
        center_lat: Center latitude.
        center_lon: Center longitude.
        num_vertices: Number of polygon vertices.
        radius_deg: Approximate radius in degrees.

    Returns:
        List of (lat, lon) vertices in clockwise order.
    """
    angles = np.sort(rng.uniform(0, 2 * np.pi, num_vertices))
    radii = rng.uniform(radius_deg * 0.6, radius_deg, num_vertices)
    vertices = []
    for angle, r in zip(angles, radii):
        lat = center_lat + r * math.cos(angle)
        lon = center_lon + r * math.sin(angle)
        vertices.append((lat, lon))
    return vertices


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
        self._polygon = _generate_polygon(rng, center_lat, center_lon)

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
                if _point_in_polygon(ac_lat, ac_lon, self._polygon):
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
        return not _point_in_polygon(state.lat, state.lon, self._polygon)

    def get_waypoint(self, agent_id: str) -> dict[str, float]:
        """Return the assigned waypoint for an agent."""
        return self._waypoints[agent_id]
