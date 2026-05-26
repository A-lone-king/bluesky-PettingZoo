"""Merge scenario renderer — FAF convergence visualization."""

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


class MergeRenderer(BaseRenderer):
    """Renderer for Merge scenario.

    Highlights the controllable aircraft (green) vs background traffic (gray),
    and draws the FAF convergence point.
    """

    def __init__(self, width: int = 800, height: int = 600) -> None:
        super().__init__(width=width, height=height, caption="Merge")
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

        controllable = info.get("controllable", []) if info else []
        background = info.get("background", []) if info else []

        # Draw background aircraft first (semi-transparent gray)
        for acid in background:
            if acid not in states:
                continue
            state = states[acid]
            x, y = latlon_to_pixel(state.lat, state.lon, bounds, self._width, self._height)
            draw_aircraft(self._screen, x, y, state.hdg, color=(120, 120, 120))
            draw_nmac_circle(self._screen, x, y)

        # Draw controllable aircraft (green, on top)
        for acid in controllable:
            if acid not in states:
                continue
            state = states[acid]
            x, y = latlon_to_pixel(state.lat, state.lon, bounds, self._width, self._height)
            draw_aircraft(self._screen, x, y, state.hdg, color=(0, 255, 0))
            draw_nmac_circle(self._screen, x, y)

        # Draw FAF waypoint (shared convergence point)
        if waypoints and isinstance(waypoints, dict):
            # Use first waypoint as FAF
            first_wp = next(iter(waypoints.values()), None)
            if first_wp and isinstance(first_wp, dict) and "lat" in first_wp:
                fx, fy = latlon_to_pixel(first_wp["lat"], first_wp["lon"], bounds, self._width, self._height)
                draw_waypoint(self._screen, fx, fy, color=(255, 100, 0), radius=8)
                # Draw FAF label
                if self._font is not None:
                    faf_text = self._font.render("FAF", True, (255, 100, 0))
                    self._screen.blit(faf_text, (fx + 10, fy - 10))

        self._draw_hud(step=step, info=info)
        self.flip()
