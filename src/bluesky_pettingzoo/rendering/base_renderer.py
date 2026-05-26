"""Base renderer — Pygame display management and HUD drawing."""

from __future__ import annotations

from typing import Any

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]


class BaseRenderer:
    """Base class for scenario-specific renderers.

    Manages Pygame initialization, screen creation, HUD overlay,
    and resource cleanup. Subclasses implement ``render_frame()``.

    Args:
        width: Window width in pixels.
        height: Window height in pixels.
        caption: Window title.
        fps: Target frames per second.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        caption: str = "BlueSky MARL",
        fps: int = 30,
    ) -> None:
        self._width = width
        self._height = height
        self._caption = caption
        self._fps = fps
        self._screen: Any = None
        self._clock: Any = None
        self._font: Any = None
        self._initialized = False

    def display(self) -> None:
        """Initialize Pygame and create the display window.

        Raises:
            ImportError: If pygame is not installed.
        """
        if pygame is None:
            raise ImportError(
                "pygame is required for rendering. "
                "Install with: pip install pygame"
            )
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
        """Render a single frame. Override in subclasses.

        Args:
            states: Aircraft states keyed by agent ID.
            waypoints: Goal waypoints keyed by agent ID.
            step: Current step number.
            info: Additional info dict.
        """
        if not self._initialized:
            return

    def _draw_hud(self, step: int = 0, info: dict[str, Any] | None = None) -> None:
        """Draw HUD overlay with step counter and info text.

        Args:
            step: Current step number.
            info: Optional info dict to display.
        """
        if not self._initialized or self._font is None:
            return

        y_offset = 10
        step_text = self._font.render(f"Step: {step}", True, (255, 255, 255))
        self._screen.blit(step_text, (10, y_offset))
        y_offset += 25

        if info:
            for key, value in info.items():
                text = self._font.render(f"{key}: {value}", True, (200, 200, 200))
                self._screen.blit(text, (10, y_offset))
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
