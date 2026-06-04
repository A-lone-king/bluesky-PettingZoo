"""Tests for BaseRenderer — rendering base class."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestBaseRendererImport:
    """Verify BaseRenderer can be imported and has required methods."""

    def test_import(self):
        from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer

        assert BaseRenderer is not None

    def test_has_display_method(self):
        from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer

        assert hasattr(BaseRenderer, "display")

    def test_has_close_method(self):
        from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer

        assert hasattr(BaseRenderer, "close")

    def test_has_draw_hud_method(self):
        from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer

        assert hasattr(BaseRenderer, "_draw_hud")


class TestBaseRendererLifecycle:
    """Verify Pygame initialization and cleanup."""

    @patch("bluesky_pettingzoo.rendering.base_renderer.pygame")
    def test_display_initializes_pygame(self, mock_pygame):
        mock_screen = MagicMock()
        mock_pygame.display.set_mode.return_value = mock_screen
        mock_pygame.Surface = MagicMock
        mock_pygame.font.Font.return_value = MagicMock()
        mock_pygame.time.Clock.return_value = MagicMock()

        from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer

        renderer = BaseRenderer(width=800, height=600)
        renderer.display()

        mock_pygame.display.set_mode.assert_called_once()
        mock_pygame.display.set_caption.assert_called_once()
        renderer.close()

    @patch("bluesky_pettingzoo.rendering.base_renderer.pygame")
    def test_close_quits_pygame(self, mock_pygame):
        mock_screen = MagicMock()
        mock_pygame.display.set_mode.return_value = mock_screen
        mock_pygame.Surface = MagicMock
        mock_pygame.font.Font.return_value = MagicMock()
        mock_pygame.time.Clock.return_value = MagicMock()

        from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer

        renderer = BaseRenderer(width=800, height=600)
        renderer.display()
        renderer.close()

        mock_pygame.quit.assert_called_once()

    @patch("bluesky_pettingzoo.rendering.base_renderer.pygame")
    def test_display_sets_screen_size(self, mock_pygame):
        mock_screen = MagicMock()
        mock_pygame.display.set_mode.return_value = mock_screen
        mock_pygame.Surface = MagicMock
        mock_pygame.font.Font.return_value = MagicMock()
        mock_pygame.time.Clock.return_value = MagicMock()

        from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer

        renderer = BaseRenderer(width=1024, height=768)
        renderer.display()

        mock_pygame.display.set_mode.assert_called_once_with((1024, 768))
        renderer.close()


class TestBaseRendererDrawHud:
    """Verify HUD drawing."""

    @patch("bluesky_pettingzoo.rendering.base_renderer.pygame")
    def test_draw_hud_renders_text(self, mock_pygame):
        mock_screen = MagicMock()
        mock_pygame.display.set_mode.return_value = mock_screen
        mock_pygame.Surface = MagicMock
        mock_font = MagicMock()
        mock_pygame.font.Font.return_value = mock_font
        mock_pygame.time.Clock.return_value = MagicMock()

        from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer

        renderer = BaseRenderer(width=800, height=600)
        renderer.display()
        renderer._draw_hud(step=10, info={"test": "value"})

        # HUD should render some text
        mock_font.render.assert_called()
        renderer.close()
