"""VerticalCR scenario renderer — Pygame visualization for vertical conflict resolution."""

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


class VerticalCRRenderer(BaseRenderer):
    """Renderer for VerticalCR scenario.

    Draws aircraft as rotated triangles, waypoints as circles,
    NMAC circles, and altitude labels for vertical separation visualization.
    """

    def __init__(self, width: int = 800, height: int = 600) -> None:
        super().__init__(width=width, height=height, caption="VerticalCR")
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
        """Render aircraft, waypoints, NMAC circles, and altitude labels.

        Args:
            states: Aircraft states keyed by agent ID.
            waypoints: Goal waypoints keyed by agent ID.
            step: Current step number.
            info: Additional info dict.
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

        for acid, state in states.items():
            x, y = latlon_to_pixel(
                state.lat, state.lon, bounds, self._width, self._height
            )
            draw_aircraft(self._screen, x, y, state.hdg)
            draw_nmac_circle(self._screen, x, y)
            # Draw altitude label
            if self._font is not None:
                alt_text = self._font.render(f"{int(state.alt)}ft", True, (255, 255, 255))
                self._screen.blit(alt_text, (x + 12, y - 8))

        if waypoints:
            for acid, wp in waypoints.items():
                wx, wy = latlon_to_pixel(
                    wp["lat"], wp["lon"], bounds, self._width, self._height
                )
                draw_waypoint(self._screen, wx, wy)

        self._draw_hud(step=step, info=info)
        self.flip()
