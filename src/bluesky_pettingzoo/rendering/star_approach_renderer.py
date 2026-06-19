"""StarApproach scenario renderer — Pygame visualization for STAR procedures.

Renders STAR waypoint sequences, approach paths, and runway threshold
for terminal airspace operations.
"""

from __future__ import annotations

from typing import Any

from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer
from bluesky_pettingzoo.rendering.common import (
    draw_aircraft,
    draw_nmac_circle,
    draw_waypoint,
    latlon_to_pixel,
)

# STAR procedure colors (one per procedure)
STAR_COLORS = {
    "ARTIP3C": (0, 200, 255),  # Cyan
    "RIVER4M": (255, 150, 0),  # Orange
    "SOBTU3G": (0, 255, 100),  # Green
}


class StarApproachRenderer(BaseRenderer):
    """Renderer for StarApproach scenario.

    Draws STAR procedures with colored waypoint sequences,
    approach paths, runway threshold, and aircraft positions.
    """

    def __init__(self, width: int = 800, height: int = 600) -> None:
        super().__init__(width=width, height=height, caption="StarApproach")
        self._bounds: dict[str, float] = {}
        self._star_procedures: dict[str, Any] = {}
        self._runway_threshold: dict[str, float] = {}

    def set_bounds(self, bounds: dict[str, float]) -> None:
        """Set the lat/lon bounds for coordinate conversion."""
        self._bounds = bounds

    def set_star_procedures(self, procedures: dict[str, Any]) -> None:
        """Set STAR procedure definitions for rendering."""
        self._star_procedures = procedures

    def set_runway_threshold(self, threshold: dict[str, float]) -> None:
        """Set runway threshold position."""
        self._runway_threshold = threshold

    def render_frame(
        self,
        states: dict[str, Any],
        waypoints: dict[str, dict[str, float]] | None = None,
        step: int = 0,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Render a single frame of the STAR approach scenario."""
        if not self._initialized or self._screen is None:
            return

        self._screen.fill((10, 10, 30))  # Dark blue background

        bounds = self._bounds or {
            "lat_min": 51.0,
            "lat_max": 54.0,
            "lon_min": 3.5,
            "lon_max": 6.5,
        }

        # Draw STAR procedures
        self._draw_star_procedures(bounds)

        # Draw runway threshold
        self._draw_runway(bounds)

        # Draw aircraft
        for acid, state in states.items():
            x, y = latlon_to_pixel(state.lat, state.lon, bounds, self._width, self._height)
            draw_aircraft(self._screen, x, y, state.hdg)
            draw_nmac_circle(self._screen, x, y)

        # Draw waypoints
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

        self._draw_hud(step=step, info=info)
        self.flip()

    def _draw_star_procedures(self, bounds: dict[str, float]) -> None:
        """Draw STAR procedure waypoint sequences."""
        for star_name, star_data in self._star_procedures.items():
            color = STAR_COLORS.get(star_name, (200, 200, 200))
            waypoints = star_data.get("waypoints", [])

            if len(waypoints) < 2:
                continue

            # Draw lines between waypoints
            for i in range(len(waypoints) - 1):
                wp1 = waypoints[i]
                wp2 = waypoints[i + 1]

                x1, y1 = latlon_to_pixel(wp1[1], wp1[2], bounds, self._width, self._height)
                x2, y2 = latlon_to_pixel(wp2[1], wp2[2], bounds, self._width, self._height)

                import pygame

                pygame.draw.line(self._screen, color, (x1, y1), (x2, y2), 2)

            # Draw waypoint labels
            for wp in waypoints:
                x, y = latlon_to_pixel(wp[1], wp[2], bounds, self._width, self._height)
                self._draw_label(wp[0], x + 10, y - 5, color)

    def _draw_runway(self, bounds: dict[str, float]) -> None:
        """Draw runway threshold."""
        if not self._runway_threshold:
            return

        lat = self._runway_threshold.get("lat", 52.3080)
        lon = self._runway_threshold.get("lon", 4.7639)
        heading = self._runway_threshold.get("heading", 270.0)

        x, y = latlon_to_pixel(lat, lon, bounds, self._width, self._height)

        import pygame

        # Draw runway as a white rectangle
        runway_length = 30
        runway_width = 6

        import math

        angle = math.radians(heading)
        dx = runway_length * math.cos(angle)
        dy = runway_length * math.sin(angle)

        start = (int(x - dx / 2), int(y - dy / 2))
        end = (int(x + dx / 2), int(y + dy / 2))

        pygame.draw.line(self._screen, (255, 255, 255), start, end, runway_width)
        self._draw_label("RWY 27", x + 15, y, (255, 255, 255))

    def _draw_label(self, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        """Draw text label."""
        if self._screen is None:
            return
        try:
            import pygame

            font = pygame.font.SysFont("Arial", 10)
            surface = font.render(text, True, color)
            self._screen.blit(surface, (x, y))
        except Exception:
            pass
