"""Route navigation scenario with crossing routes.

Aircraft follow predefined routes through a waypoint network.
Some routes cross, creating natural conflict opportunities.
Used to test route-aware conflict detection and resolution.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.geometry import point_at_distance
from bluesky_pettingzoo.utils.types import ConflictConfig, Route, SpawnConfig

CRUISE_ALT_FT = 35000.0
CRUISE_SPEED_KT = 450.0
ROUTE_WAYPOINT_DISTANCE_NM = 40.0


class RouteNavScenario(BaseScenario):
    """Navigation scenario with crossing routes.

    Defines a waypoint network with 5-7 waypoints.  Each aircraft is
    assigned a route (2-4 waypoints).  Routes share some waypoints or
    cross near them, creating natural conflict opportunities.

    Args:
        num_aircraft: Number of aircraft (and routes) to spawn.
        seed: Optional seed for reproducibility.
    """

    def __init__(
        self,
        num_aircraft: int = 3,
        num_aircraft_range: tuple[int, int] | None = None,
        seed: int | None = None,
    ) -> None:
        self._num_aircraft = num_aircraft
        self._num_aircraft_range = num_aircraft_range
        self._seed = seed
        self._agents: list[str] = []
        self._routes: dict[str, Route] = {}
        self._goals: dict[str, dict[str, float]] = {}
        self._bounds: dict[str, float] = {}

    @property
    def action_dimensions(self) -> list[int]:
        return [0, 2]  # heading + speed

    @property
    def num_aircraft_range(self) -> tuple[int, int] | None:
        """Return dynamic aircraft count range if configured."""
        return self._num_aircraft_range

    def reset(self, rng: np.random.RandomState) -> None:
        """Randomize aircraft count for procedural generation."""
        if self._num_aircraft_range is not None:
            self._num_aircraft = int(rng.randint(
                self._num_aircraft_range[0],
                self._num_aircraft_range[1] + 1,
            ))
        self._agents = []
        self._routes = {}
        self._goals = {}

    def setup(
        self,
        rng: np.random.RandomState,
        airspace_bounds: dict[str, float],
    ) -> list[str]:
        """Initialize scenario: build waypoint network and assign routes."""
        self._bounds = airspace_bounds
        self._agents = [f"AC{i:03d}" for i in range(self._num_aircraft)]
        self._routes = {}
        self._goals = {}

        # Generate a cluster of waypoints around the center
        center_lat = (airspace_bounds["lat_min"] + airspace_bounds["lat_max"]) / 2
        center_lon = (airspace_bounds["lon_min"] + airspace_bounds["lon_max"]) / 2

        num_waypoints = max(5, self._num_aircraft + 2)
        waypoints: list[dict[str, float]] = []
        for i in range(num_waypoints):
            angle = rng.uniform(0, 360)
            dist = rng.uniform(15, 50)
            wp_lat, wp_lon = point_at_distance(center_lat, center_lon, dist, angle)
            waypoints.append({"lat": wp_lat, "lon": wp_lon})

        # Assign each aircraft a route: pick 2-4 sequential waypoints
        for idx, acid in enumerate(self._agents):
            # Start from a different waypoint for each aircraft
            start_idx = idx % len(waypoints)
            route_len = min(3, len(waypoints))
            route_wps = []
            for j in range(route_len):
                wp = waypoints[(start_idx + j) % len(waypoints)]
                route_wps.append({"lat": wp["lat"], "lon": wp["lon"]})

            self._routes[acid] = Route(waypoints=route_wps)

            # Goal = last waypoint of route
            goal = route_wps[-1]
            self._goals[acid] = {
                "lat": goal["lat"],
                "lon": goal["lon"],
                "alt": CRUISE_ALT_FT,
                "hdg": 0.0,
            }

        return list(self._agents)

    def get_spawn_config(self) -> SpawnConfig:
        return SpawnConfig(
            altitude_range=(CRUISE_ALT_FT, CRUISE_ALT_FT),
            speed_range=(400, 500),
            heading_range=(0, 360),
        )

    def get_conflict_config(self) -> ConflictConfig:
        return ConflictConfig(
            nmac_horizontal_nm=5.0,
            nmac_vertical_ft=1000.0,
            warning_horizontal_nm=10.0,
            warning_vertical_ft=2000.0,
        )

    def get_waypoint(self, agent_id: str) -> dict[str, float]:
        return self._goals[agent_id]

    def get_goal(self, agent_id: str) -> dict[str, float]:
        return self._goals[agent_id]

    def get_route(self, agent_id: str) -> Route:
        """Get the route assigned to an agent."""
        return self._routes[agent_id]

    def get_routes(self) -> dict[str, Route]:
        """Get all routes keyed by agent ID."""
        return dict(self._routes)

    def configure_npc_navigation(self, wrapper: Any) -> list[str]:
        """Configure LNAV for all aircraft to follow their assigned routes.

        Uses BlueSky LNAV to make aircraft automatically follow waypoints.

        Args:
            wrapper: BlueSkyWrapper instance for sending commands.

        Returns:
            List of BlueSky commands sent.
        """

        commands: list[str] = []
        wrapper.send_command("reso off")
        commands.append("reso off")

        for acid in self._agents:
            route = self._routes.get(acid)
            if route is None or not route.waypoints:
                continue
            # Set first waypoint as destination, enable LNAV
            first_wp = route.waypoints[0]
            wrapper.set_destination(acid, first_wp["lat"], first_wp["lon"])
            commands.append(f"DEST {acid} {first_wp['lat']} {first_wp['lon']}")
            wrapper.enable_lnav(acid)
            commands.append(f"LNAV {acid} ON")

        return commands
