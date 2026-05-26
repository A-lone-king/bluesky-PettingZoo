"""StaticObstacle scenario renderer — no-fly zone polygon visualization."""

from __future__ import annotations

from typing import Any

from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer
from bluesky_pettingzoo.rendering.common import (
    draw_aircraft,
    draw_nmac_circle,
    draw_sector_polygon,
    draw_waypoint,
    latlon_to_pixel,
)


class StaticObstacleRenderer(BaseRenderer):
    """Renderer for StaticObstacle scenario.

    Draws obstacle polygons as no-fly zones with red outlines,
    aircraft, and waypoints.
    """

    def __init__(self, width: int = 800, height: int = 600) -> None:
        super().__init__(width=width, height=height, caption="StaticObstacle")
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
        if not self._initialized or self._screen is None:
            return

        self._screen.fill((0, 0, 0))
        bounds = self._bounds or {
            "lat_min": 39.0, "lat_max": 41.0,
            "lon_min": 116.0, "lon_max": 118.0,
        }

        # Draw obstacle polygons (red no-fly zones)
        obstacles = info.get("obstacles", []) if info else []
        for polygon in obstacles:
            if polygon:
                draw_sector_polygon(
                    self._screen, polygon, bounds, self._width, self._height,
                    color=(255, 50, 50), line_width=2,
                )

        for acid, state in states.items():
            x, y = latlon_to_pixel(state.lat, state.lon, bounds, self._width, self._height)
            draw_aircraft(self._screen, x, y, state.hdg)
            draw_nmac_circle(self._screen, x, y)

        if waypoints and isinstance(waypoints, dict):
            for acid, wp in waypoints.items():
                if isinstance(wp, dict) and "lat" in wp:
                    wx, wy = latlon_to_pixel(wp["lat"], wp["lon"], bounds, self._width, self._height)
                    draw_waypoint(self._screen, wx, wy)

        self._draw_hud(step=step, info=info)
        self.flip()
