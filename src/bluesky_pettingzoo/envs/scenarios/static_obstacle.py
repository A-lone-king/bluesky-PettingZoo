"""Static obstacle avoidance scenario.

Aircraft navigate to waypoints while avoiding randomly generated
no-fly (restricted) zones.  Entering an obstacle terminates the
episode for that agent.
"""

from __future__ import annotations

import math

import numpy as np

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.geometry import (
    point_at_distance,
    point_in_polygon,
)
from bluesky_pettingzoo.utils.types import AircraftState, ConflictConfig, SpawnConfig

# Constants (matching bluesky-gym reference)
NUM_OBSTACLES = 10
OBSTACLE_DISTANCE_MIN_KM = 20
OBSTACLE_DISTANCE_MAX_KM = 150
OBSTACLE_AREA_MIN_NM2 = 50
OBSTACLE_AREA_MAX_NM2 = 1000
CRUISE_ALT_FT = 35000.0
SPEED_KT = 150.0
WAYPOINT_DISTANCE_MIN_NM = 80
WAYPOINT_DISTANCE_MAX_NM = 150
KM_TO_NM = 0.539957  # 1 km = 0.539957 NM
NM_TO_DEG_LAT = 1.0 / 60.0  # 1 NM ≈ 1/60 degree latitude


def _nm_to_deg_lon(nm: float, lat: float) -> float:
    """Convert nautical miles to degrees longitude at a given latitude."""
    return nm / (60.0 * max(math.cos(math.radians(lat)), 1e-6))


def _generate_obstacle_polygon(
    rng: np.random.RandomState,
    center_lat: float,
    center_lon: float,
) -> list[tuple[float, float]]:
    """Generate a random irregular polygon around a center point.

    The polygon has 3+ vertices, with area in [OBSTACLE_AREA_MIN_NM2,
    OBSTACLE_AREA_MAX_NM2] NM².  Vertices are placed on a circumscribing
    circle and additional vertices are added if the area is too small.

    Args:
        rng: Random number generator.
        center_lat: Center latitude in degrees.
        center_lon: Center longitude in degrees.

    Returns:
        List of (lat, lon) vertices.
    """
    # Sample area and compute circumscribing circle radius (in NM)
    poly_area = rng.randint(OBSTACLE_AREA_MIN_NM2 * 2, OBSTACLE_AREA_MAX_NM2)
    R_nm = math.sqrt(poly_area / math.pi)

    # Generate 3 random angles on the circle, sorted clockwise
    angles = np.sort(rng.uniform(0, 2 * np.pi, 3))

    # Convert to lat/lon offsets
    vertices: list[tuple[float, float]] = []
    for angle in angles:
        dlat = R_nm * math.cos(angle) * NM_TO_DEG_LAT
        dlon = R_nm * math.sin(angle) * _nm_to_deg_lon(R_nm, center_lat)
        vertices.append((center_lat + dlat, center_lon + dlon))

    # Add more vertices if polygon area is too small
    # Use shoelace formula to estimate area in deg², convert to NM²
    for _ in range(10):  # max 10 extra vertices
        area_nm2 = _polygon_area_nm2(vertices, center_lat)
        if area_nm2 >= OBSTACLE_AREA_MIN_NM2:
            break
        angle = rng.uniform(0, 2 * math.pi)
        dlat = R_nm * math.cos(angle) * NM_TO_DEG_LAT
        dlon = R_nm * math.sin(angle) * _nm_to_deg_lon(R_nm, center_lat)
        vertices.append((center_lat + dlat, center_lon + dlon))
        # Re-sort by angle to maintain convexity
        vertices = _sort_vertices_clockwise(vertices, center_lat, center_lon)

    return vertices


def _polygon_area_nm2(
    vertices: list[tuple[float, float]],
    ref_lat: float,
) -> float:
    """Approximate polygon area in NM² using the shoelace formula.

    Converts degree deltas to NM using a reference latitude.
    """
    n = len(vertices)
    if n < 3:
        return 0.0

    # Convert to local NM coordinates
    nm_vertices: list[tuple[float, float]] = []
    for lat, lon in vertices:
        y_nm = (lat - ref_lat) / NM_TO_DEG_LAT
        x_nm = (lon - vertices[0][1]) / _nm_to_deg_lon(1.0, ref_lat)
        nm_vertices.append((x_nm, y_nm))

    # Shoelace formula
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += nm_vertices[i][0] * nm_vertices[j][1]
        area -= nm_vertices[j][0] * nm_vertices[i][1]
    return abs(area) / 2.0


def _sort_vertices_clockwise(
    vertices: list[tuple[float, float]],
    center_lat: float,
    center_lon: float,
) -> list[tuple[float, float]]:
    """Sort vertices by angle from center, clockwise."""

    def angle(v: tuple[float, float]) -> float:
        return math.atan2(v[1] - center_lon, v[0] - center_lat)

    return sorted(vertices, key=angle)


class StaticObstacleScenario(BaseScenario):
    """Static obstacle avoidance scenario.

    Aircraft navigate to waypoints while avoiding randomly placed
    restricted-area polygons.  Entering any obstacle terminates the
    agent.  Reaching the waypoint also terminates the agent (success).

    Args:
        num_aircraft: Number of aircraft to spawn.
        num_obstacles: Number of obstacle polygons to generate.
        seed: Optional seed for reproducibility.
    """

    def __init__(
        self,
        num_aircraft: int = 1,
        num_aircraft_range: tuple[int, int] | None = None,
        num_obstacles: int = NUM_OBSTACLES,
        seed: int | None = None,
    ) -> None:
        self._num_aircraft = num_aircraft
        self._num_aircraft_range = num_aircraft_range
        self._num_obstacles = num_obstacles
        self._seed = seed
        self._agents: list[str] = []
        self._waypoints: dict[str, dict[str, float]] = {}
        self._obstacles: list[list[tuple[float, float]]] = []
        self._obstacle_names: list[str] = []
        self._initial_positions: dict[str, tuple[float, float]] | None = None
        self._bounds: dict[str, float] = {}

    @property
    def action_dimensions(self) -> list[int]:
        """Heading + speed (matching bluesky-gym StaticObstacleEnv)."""
        return [0, 2]

    @property
    def num_aircraft_range(self) -> tuple[int, int] | None:
        """Return dynamic aircraft count range if configured."""
        return self._num_aircraft_range

    def get_obstacles(self) -> list[list[tuple[float, float]]]:
        """Return obstacle polygons for the current episode."""
        return self._obstacles

    def get_obstacle_names(self) -> list[str]:
        """Return BlueSky area filter names for obstacles."""
        return self._obstacle_names

    def get_initial_positions(self) -> dict[str, tuple[float, float]] | None:
        """Return initial positions for each agent."""
        return self._initial_positions

    def setup(
        self,
        rng: np.random.RandomState,
        airspace_bounds: dict[str, float],
    ) -> list[str]:
        """Initialize scenario: generate obstacles and waypoints.

        Places aircraft at random positions, generates obstacle polygons
        around them, and assigns waypoints that are outside all obstacles.
        """
        self._bounds = airspace_bounds
        self._agents = [f"AC{i:03d}" for i in range(self._num_aircraft)]
        self._waypoints = {}
        self._obstacles = []
        self._obstacle_names = []
        self._initial_positions = {}

        center_lat = (airspace_bounds["lat_min"] + airspace_bounds["lat_max"]) / 2
        center_lon = (airspace_bounds["lon_min"] + airspace_bounds["lon_max"]) / 2

        # Place aircraft near center
        for acid in self._agents:
            ac_lat = rng.uniform(center_lat - 0.1, center_lat + 0.1)
            ac_lon = rng.uniform(center_lon - 0.1, center_lon + 0.1)
            self._initial_positions[acid] = (ac_lat, ac_lon)

        # Generate obstacle polygons around the first aircraft (reference point)
        ref_lat, ref_lon = self._initial_positions[self._agents[0]]
        for i in range(self._num_obstacles):
            dist_km = rng.uniform(OBSTACLE_DISTANCE_MIN_KM, OBSTACLE_DISTANCE_MAX_KM)
            hdg = rng.uniform(0, 360)
            dist_nm = dist_km * KM_TO_NM
            obs_lat, obs_lon = point_at_distance(ref_lat, ref_lon, dist_nm, hdg)
            polygon = _generate_obstacle_polygon(rng, obs_lat, obs_lon)
            self._obstacles.append(polygon)
            self._obstacle_names.append(f"restricted_area_{i + 1}")

        # Assign waypoints outside all obstacles
        for acid in self._agents:
            ac_lat, ac_lon = self._initial_positions[acid]
            wp_lat, wp_lon = self._find_waypoint_outside_obstacles(
                rng,
                ac_lat,
                ac_lon,
            )
            bearing = (
                math.degrees(
                    math.atan2(
                        wp_lon - ac_lon,
                        wp_lat - ac_lat,
                    )
                )
                % 360
            )
            self._waypoints[acid] = {
                "lat": wp_lat,
                "lon": wp_lon,
                "alt": CRUISE_ALT_FT,
                "hdg": bearing,
            }

        return list(self._agents)

    def _find_waypoint_outside_obstacles(
        self,
        rng: np.random.RandomState,
        ac_lat: float,
        ac_lon: float,
    ) -> tuple[float, float]:
        """Find a waypoint that is outside all obstacle polygons.

        Uses rejection sampling.  Falls back to a distant point if
        no valid waypoint is found after many attempts.
        """
        for _ in range(200):
            bearing = rng.uniform(0, 360)
            dist = rng.uniform(WAYPOINT_DISTANCE_MIN_NM, WAYPOINT_DISTANCE_MAX_NM)
            wp_lat, wp_lon = point_at_distance(ac_lat, ac_lon, dist, bearing)

            # Check that waypoint is outside all obstacles
            inside_any = False
            for polygon in self._obstacles:
                if point_in_polygon(wp_lat, wp_lon, polygon):
                    inside_any = True
                    break
            if not inside_any:
                return wp_lat, wp_lon

        # Fallback: place far away in a random direction
        bearing = rng.uniform(0, 360)
        return point_at_distance(ac_lat, ac_lon, WAYPOINT_DISTANCE_MAX_NM * 2, bearing)

    def get_spawn_config(self) -> SpawnConfig:
        """Cruise altitude, 150kt (matching bluesky-gym)."""
        return SpawnConfig(
            altitude_range=(CRUISE_ALT_FT, CRUISE_ALT_FT),
            speed_range=(SPEED_KT, SPEED_KT),
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

    def should_truncate(
        self,
        agent_id: str,
        state: AircraftState,
        airspace_bounds: dict[str, float],
    ) -> bool:
        """StaticObstacle does not use truncation.

        Obstacle intrusion triggers termination (handled by ObstacleIntrusion
        reward component in parallel_env).  Waypoint arrival also triggers
        termination.  Neither is a truncation.
        """
        return False

    def get_waypoint(self, agent_id: str) -> dict[str, float]:
        """Return the assigned waypoint for an agent."""
        return self._waypoints[agent_id]

    def reset(self, rng: np.random.RandomState) -> None:
        """Clear scenario state and randomize for procedural generation."""
        if self._num_aircraft_range is not None:
            self._num_aircraft = int(
                rng.randint(
                    self._num_aircraft_range[0],
                    self._num_aircraft_range[1] + 1,
                )
            )
        self._agents = []
        self._waypoints = {}
        self._obstacles = []
        self._obstacle_names = []
        self._initial_positions = None
