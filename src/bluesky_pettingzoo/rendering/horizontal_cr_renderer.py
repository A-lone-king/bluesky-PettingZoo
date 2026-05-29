"""HorizontalCR scenario renderer — Pygame visualization for horizontal conflict resolution."""

from __future__ import annotations

from typing import Any

from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer
from bluesky_pettingzoo.rendering.common import draw_nmac_circle, latlon_to_pixel


class HorizontalCRRenderer(BaseRenderer):
    """Renderer for HorizontalCR scenario.

    Draws aircraft as rotated triangles, waypoints as circles,
    and NMAC conflict circles around each aircraft.
    """

    def __init__(self, width: int = 800, height: int = 600) -> None:
        super().__init__(width=width, height=height, caption="HorizontalCR")

    def render_frame(
        self,
        states: dict[str, Any],
        waypoints: dict[str, dict[str, float]] | None = None,
        step: int = 0,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Render aircraft, waypoints, and NMAC circles.

        Args:
            states: Aircraft states keyed by agent ID.
            waypoints: Goal waypoints keyed by agent ID.
            step: Current step number.
            info: Additional info dict.
        """
        if not self._initialized or self._screen is None:
            return

        # Call base renderer to draw aircraft and waypoints
        super().render_frame(states, waypoints, step, info)

        # Add NMAC circles on top
        for acid, state in states.items():
            x, y = latlon_to_pixel(
                state.lat, state.lon, self._bounds, self._width, self._height
            )
            draw_nmac_circle(self._screen, x, y)

        self.flip()
