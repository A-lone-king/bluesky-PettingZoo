"""Base renderer — Pygame display management and HUD drawing."""

from __future__ import annotations

from typing import Any

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]

from bluesky_pettingzoo.rendering.common import (
    compute_pixels_per_nm,
    draw_aircraft,
    draw_hud_panel,
    draw_sky_gradient,
    draw_waypoint,
    latlon_to_pixel,
)


class BaseRenderer:
    """Base class for scenario-specific renderers.

    Manages Pygame initialization, screen creation, HUD overlay,
    and resource cleanup. Provides default render_frame implementation
    that subclasses can extend.

    Args:
        width: Window width in pixels.
        height: Window height in pixels.
        caption: Window title.
        fps: Target frames per second.
        bounds: Geographic bounds for rendering.
    """

    _DEFAULT_BOUNDS = {
        "lat_min": 39.0,
        "lat_max": 41.0,
        "lon_min": 116.0,
        "lon_max": 118.0,
    }

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        caption: str = "BlueSky MARL",
        fps: int = 30,
        bounds: dict[str, float] | None = None,
        background_color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self._width = width
        self._height = height
        self._caption = caption
        self._fps = fps
        self._screen: Any = None
        self._clock: Any = None
        self._font: Any = None
        self._initialized = False
        self._bounds = bounds or self._DEFAULT_BOUNDS.copy()
        self._background_color = background_color

    def set_bounds(self, bounds: dict[str, float]) -> None:
        """Set geographic bounds for rendering.

        Args:
            bounds: Dictionary with lat_min, lat_max, lon_min, lon_max
        """
        self._bounds = bounds

    def _compute_pixels_per_nm(self) -> float:
        """Compute pixels per nautical mile from current bounds.

        Returns:
            Pixels per NM scale factor.
        """
        return compute_pixels_per_nm(self._bounds, self._height)

    def display(self) -> None:
        """Initialize Pygame and create the display window.

        Raises:
            ImportError: If pygame is not installed.
        """
        if pygame is None:
            raise ImportError("pygame is required for rendering. Install with: pip install pygame")
        pygame.init()
        self._screen = pygame.display.set_mode((self._width, self._height))
        pygame.display.set_caption(self._caption)
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, 24)
        self._initialized = True

    def render_frame(
        self,
        states: dict[str, Any],
        waypoints: dict[str, dict[str, float]] | None = None,
        step: int = 0,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Render a single frame with aircraft and waypoints.

        Subclasses can override to add custom rendering before/after
        calling super().render_frame().

        Args:
            states: Aircraft states keyed by agent ID.
            waypoints: Goal waypoints keyed by agent ID.
            step: Current step number.
            info: Additional info dict.
        """
        if not self._initialized:
            return

        # Clear screen with sky gradient
        draw_sky_gradient(
            self._screen,
            self._width,
            self._height,
            top_color=(135, 206, 235),
            bottom_color=(176, 224, 230),
        )

        # Draw aircraft
        for agent_id, state in states.items():
            x, y = latlon_to_pixel(state.lat, state.lon, self._bounds, self._width, self._height)
            draw_aircraft(self._screen, x, y, state.hdg)

        # Draw waypoints
        if waypoints:
            for agent_id, wp in waypoints.items():
                wx, wy = latlon_to_pixel(
                    wp["lat"], wp["lon"], self._bounds, self._width, self._height
                )
                draw_waypoint(self._screen, wx, wy)

        # Draw HUD
        self._draw_hud(step, info)

    def _draw_hud(self, step: int = 0, info: dict[str, Any] | None = None) -> None:
        """Draw HUD overlay with step counter and info text.

        Args:
            step: Current step number.
            info: Optional info dict to display.
        """
        if not self._initialized or self._font is None:
            return

        # Draw semi-transparent HUD panel
        num_lines = 1 + (len(info) if info else 0)
        panel_height = 30 + num_lines * 25
        draw_hud_panel(
            self._screen,
            x=10,
            y=10,
            width=220,
            height=panel_height,
            color=(0, 0, 0),
            alpha=128,
        )

        y_offset = 20
        step_text = self._font.render(f"Step: {step}", True, (255, 255, 255))
        self._screen.blit(step_text, (20, y_offset))
        y_offset += 25

        if info:
            for key, value in info.items():
                text = self._font.render(f"{key}: {value}", True, (200, 200, 200))
                self._screen.blit(text, (20, y_offset))
                y_offset += 25

    def flip(self) -> None:
        """Flip the display and tick the clock."""
        if self._initialized and self._screen is not None:
            pygame.display.flip()
            self._clock.tick(self._fps)

    def close(self) -> None:
        """Clean up Pygame resources."""
        if self._initialized:
            pygame.quit()
            self._initialized = False
