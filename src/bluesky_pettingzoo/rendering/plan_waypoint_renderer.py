"""PlanWaypoint scenario renderer — ordered waypoint chain visualization."""

from __future__ import annotations

from typing import Any

from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer
from bluesky_pettingzoo.rendering.common import (
    draw_aircraft,
    draw_nmac_circle,
    draw_sky_gradient,
    draw_waypoint_circle,
    latlon_to_pixel,
)

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]


class PlanWaypointRenderer(BaseRenderer):
    """Renderer for PlanWaypoint scenario.

    Draws an ordered chain of waypoints with reached (green) vs pending (white)
    color coding, connected by lines.
    """

    def __init__(self, width: int = 800, height: int = 600) -> None:
        super().__init__(width=width, height=height, caption="PlanWaypoint")
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
        """Render one frame showing aircraft positions and waypoint paths.

        Args:
            states: Aircraft states keyed by agent ID.
            waypoints: Optional goal waypoints keyed by agent ID.
            step: Current simulation step number.
            info: Optional additional info dict.
        """
        if not self._initialized or self._screen is None:
            return

        # Draw sky gradient background
        draw_sky_gradient(self._screen, self._width, self._height)

        bounds = self._bounds or {
            "lat_min": 39.0,
            "lat_max": 41.0,
            "lon_min": 116.0,
            "lon_max": 118.0,
        }

        for acid, state in states.items():
            x, y = latlon_to_pixel(state.lat, state.lon, bounds, self._width, self._height)
            draw_aircraft(self._screen, x, y, state.hdg)
            draw_nmac_circle(self._screen, x, y)

        # Draw ordered waypoint chain
        if waypoints and isinstance(waypoints, list) and pygame is not None:
            pixel_points = []
            for wp in waypoints:
                wx, wy = latlon_to_pixel(wp["lat"], wp["lon"], bounds, self._width, self._height)
                pixel_points.append((wx, wy))
                reached = wp.get("reached", False)
                # Use bluesky-gym style concentric circles for waypoints
                if reached:
                    draw_waypoint_circle(
                        self._screen,
                        wx,
                        wy,
                        outer_color=(0, 200, 0),
                        inner_color=(0, 150, 0),
                    )
                else:
                    draw_waypoint_circle(
                        self._screen,
                        wx,
                        wy,
                        outer_color=(255, 255, 255),
                        inner_color=(135, 206, 235),
                    )

            # Connect waypoints with lines
            if len(pixel_points) >= 2:
                pygame.draw.lines(self._screen, (100, 100, 100), False, pixel_points, 1)

        self._draw_hud(step=step, info=info)
        self.flip()
