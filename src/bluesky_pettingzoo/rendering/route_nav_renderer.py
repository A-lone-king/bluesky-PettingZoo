"""RouteNav scenario renderer — multi-waypoint route visualization."""

from __future__ import annotations

from typing import Any

from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer
from bluesky_pettingzoo.rendering.common import (
    draw_aircraft,
    draw_nmac_circle,
    draw_waypoint,
    latlon_to_pixel,
)

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]


class RouteNavRenderer(BaseRenderer):
    """Renderer for RouteNav scenario.

    Draws each aircraft's full route as a connected line of waypoints,
    with the goal waypoint highlighted.
    """

    def __init__(self, width: int = 800, height: int = 600) -> None:
        super().__init__(width=width, height=height, caption="RouteNav")
        self._bounds: dict[str, float] = {}

    def set_bounds(self, bounds: dict[str, float]) -> None:
        """Set the lat/lon bounds for coordinate conversion."""
        self._bounds = bounds

    def render_frame(
        self,
        states: dict[str, Any],
        waypoints: dict[str, dict[str, float]] | None = None,
        step: int = 0,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Render one frame showing aircraft positions and route networks.

        Args:
            states: Aircraft states keyed by agent ID.
            waypoints: Optional goal waypoints keyed by agent ID.
            step: Current simulation step number.
            info: Optional additional info dict.
        """
        if not self._initialized or self._screen is None:
            return

        self._screen.fill((0, 0, 0))
        bounds = self._bounds or {
            "lat_min": 39.0,
            "lat_max": 41.0,
            "lon_min": 116.0,
            "lon_max": 118.0,
        }

        # Draw route lines for each aircraft
        routes = info.get("routes", {}) if info else {}
        if routes and pygame is not None:
            for acid, route in routes.items():
                route_wps = route.get("waypoints", []) if isinstance(route, dict) else []
                if len(route_wps) >= 2:
                    pixel_points = [
                        latlon_to_pixel(wp["lat"], wp["lon"], bounds, self._width, self._height)
                        for wp in route_wps
                    ]
                    pygame.draw.lines(self._screen, (80, 80, 80), False, pixel_points, 1)
                    # Draw intermediate waypoints as small dots
                    for px, py in pixel_points[:-1]:
                        draw_waypoint(self._screen, px, py, color=(150, 150, 150), radius=3)
                    # Draw goal waypoint (last) prominently
                    gx, gy = pixel_points[-1]
                    draw_waypoint(self._screen, gx, gy, color=(255, 255, 0), radius=6)

        for acid, state in states.items():
            x, y = latlon_to_pixel(state.lat, state.lon, bounds, self._width, self._height)
            draw_aircraft(self._screen, x, y, state.hdg)
            draw_nmac_circle(self._screen, x, y)

        # Draw goal waypoints from waypoints dict
        if waypoints and isinstance(waypoints, dict):
            for acid, wp in waypoints.items():
                if isinstance(wp, dict) and "lat" in wp:
                    wx, wy = latlon_to_pixel(
                        wp["lat"],
                        wp["lon"],
                        bounds,
                        self._width,
                        self._height,
                    )
                    draw_waypoint(self._screen, wx, wy, color=(255, 255, 0), radius=6)

        self._draw_hud(step=step, info=info)
        self.flip()
