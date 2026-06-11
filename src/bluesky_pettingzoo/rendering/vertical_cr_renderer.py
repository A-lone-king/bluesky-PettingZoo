"""VerticalCR scenario renderer — aviation-style visualization with altitude labels."""

from __future__ import annotations

from typing import Any

from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer
from bluesky_pettingzoo.rendering.common import (
    draw_aircraft_dot,
    draw_heading_line,
    draw_nmac_circle,
    draw_sky_gradient,
    draw_waypoint,
    latlon_to_pixel,
)
from bluesky_pettingzoo.utils.geometry import haversine_distance

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]

# Aviation-style color palette
_SKY_BLUE = (135, 206, 235)
_AIRCRAFT_COLOR = (30, 30, 30)
_PROTECTION_COLOR = (30, 30, 30)
_CONFLICT_COLOR = (255, 0, 0)
_HEADING_COLOR = (80, 80, 80)
_ALT_LABEL_COLOR = (255, 255, 255)

# Separation thresholds (NM)
_NMAC_HORIZONTAL_NM = 5.0


class VerticalCRRenderer(BaseRenderer):
    """Renderer for VerticalCR scenario.

    Aviation-style rendering with light blue background, black protection
    zones, heading lines, altitude labels, and red conflict circles.
    """

    def __init__(self, width: int = 800, height: int = 600) -> None:
        super().__init__(
            width=width,
            height=height,
            caption="VerticalCR",
            background_color=_SKY_BLUE,
        )

    def render_frame(
        self,
        states: dict[str, Any],
        waypoints: dict[str, dict[str, float]] | None = None,
        step: int = 0,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Render aircraft with protection zones, heading lines, altitude labels, and conflicts.

        Args:
            states: Aircraft states keyed by agent ID.
            waypoints: Goal waypoints keyed by agent ID.
            step: Current step number.
            info: Additional info dict.
        """
        if not self._initialized or self._screen is None:
            return

        # Draw sky gradient background
        draw_sky_gradient(self._screen, self._width, self._height)
        ppm = self._compute_pixels_per_nm()

        # Detect conflicting aircraft pairs
        conflict_ids = self._detect_conflicts(states)

        # Draw aircraft with protection zones, heading lines, and altitude labels
        for acid, state in states.items():
            x, y = latlon_to_pixel(state.lat, state.lon, self._bounds, self._width, self._height)
            # Black protection zone
            draw_nmac_circle(
                self._screen,
                x,
                y,
                radius_nm=_NMAC_HORIZONTAL_NM,
                pixels_per_nm=ppm,
                color=_PROTECTION_COLOR,
                width=1,
            )
            # Red conflict circle
            if acid in conflict_ids:
                draw_nmac_circle(
                    self._screen,
                    x,
                    y,
                    radius_nm=_NMAC_HORIZONTAL_NM,
                    pixels_per_nm=ppm,
                    color=_CONFLICT_COLOR,
                    width=2,
                )
            # Heading line
            draw_heading_line(
                self._screen,
                x,
                y,
                state.hdg,
                length=30,
                color=_HEADING_COLOR,
                width=1,
            )
            # Aircraft dot
            draw_aircraft_dot(self._screen, x, y, color=_AIRCRAFT_COLOR, size=4)
            # Altitude label
            if self._font is not None:
                alt_text = self._font.render(
                    f"{int(state.alt)}ft",
                    True,
                    _ALT_LABEL_COLOR,
                )
                self._screen.blit(alt_text, (x + 12, y - 8))

        # Draw waypoints
        if waypoints:
            for acid, wp in waypoints.items():
                wx, wy = latlon_to_pixel(
                    wp["lat"], wp["lon"], self._bounds, self._width, self._height
                )
                draw_waypoint(self._screen, wx, wy)

        self._draw_hud(step=step, info=info)
        self.flip()

    def _detect_conflicts(self, states: dict[str, Any]) -> set[str]:
        """Detect aircraft pairs within NMAC horizontal separation.

        Args:
            states: Aircraft states keyed by agent ID.

        Returns:
            Set of aircraft IDs involved in conflicts.
        """
        conflict_ids: list[str] = []
        acid_list = list(states.keys())
        for i in range(len(acid_list)):
            for j in range(i + 1, len(acid_list)):
                s1 = states[acid_list[i]]
                s2 = states[acid_list[j]]
                dist = haversine_distance(s1.lat, s1.lon, s2.lat, s2.lon)
                if dist < _NMAC_HORIZONTAL_NM:
                    conflict_ids.append(acid_list[i])
                    conflict_ids.append(acid_list[j])
        return set(conflict_ids)
