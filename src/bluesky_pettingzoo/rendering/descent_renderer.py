"""Descent scenario renderer — altitude-focused visualization."""

from __future__ import annotations

from typing import Any

from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer
from bluesky_pettingzoo.rendering.common import (
    draw_aircraft,
    draw_ground,
    draw_nmac_circle,
    draw_runway,
    draw_sky_gradient,
    draw_waypoint,
    latlon_to_pixel,
)

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]


class DescentRenderer(BaseRenderer):
    """Renderer for Descent scenario.

    Draws aircraft with altitude labels and a runway marker at the center.
    Emphasizes the vertical dimension of the descent profile.
    """

    def __init__(self, width: int = 800, height: int = 600) -> None:
        super().__init__(width=width, height=height, caption="Descent")
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
        """Render one frame showing aircraft positions and descent paths.

        Args:
            states: Aircraft states keyed by agent ID.
            waypoints: Optional goal waypoints keyed by agent ID.
            step: Current simulation step number.
            info: Optional additional info dict.
        """
        if not self._initialized or self._screen is None:
            return

        bounds = self._bounds or {
            "lat_min": 39.0,
            "lat_max": 41.0,
            "lon_min": 116.0,
            "lon_max": 118.0,
        }

        # Draw sky gradient background
        draw_sky_gradient(self._screen, self._width, self._height)

        # Draw ground
        draw_ground(self._screen, self._width, self._height, ground_ratio=0.2)

        # Draw runway
        draw_runway(
            self._screen,
            x=self._width // 2,
            y=int(self._height * 0.85),
            length=200,
            width=15,
        )

        for acid, state in states.items():
            x, y = latlon_to_pixel(state.lat, state.lon, bounds, self._width, self._height)
            draw_aircraft(self._screen, x, y, state.hdg)
            draw_nmac_circle(self._screen, x, y)
            # Draw altitude label
            if self._font is not None:
                alt_text = self._font.render(f"{int(state.alt)}ft", True, (255, 255, 255))
                self._screen.blit(alt_text, (x + 12, y - 8))

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
                    draw_waypoint(self._screen, wx, wy)
                    # Draw target altitude at waypoint
                    if self._font is not None and "alt" in wp:
                        tgt_text = self._font.render(f"T:{int(wp['alt'])}ft", True, (200, 200, 0))
                        self._screen.blit(tgt_text, (wx + 8, wy - 8))

        self._draw_hud(step=step, info=info)
        self.flip()
