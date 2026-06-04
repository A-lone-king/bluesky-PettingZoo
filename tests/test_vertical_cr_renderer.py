"""Tests for VerticalCRRenderer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestVerticalCRRendererImport:
    """Verify VerticalCRRenderer can be imported."""

    def test_import(self):
        from bluesky_pettingzoo.rendering.vertical_cr_renderer import VerticalCRRenderer

        assert VerticalCRRenderer is not None

    def test_inherits_base(self):
        from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer
        from bluesky_pettingzoo.rendering.vertical_cr_renderer import VerticalCRRenderer

        assert issubclass(VerticalCRRenderer, BaseRenderer)


class TestVerticalCRRendererFrame:
    """Verify VerticalCR renders aircraft dots, protection zones, heading lines, and altitude labels."""  # noqa: E501

    @patch(
        "bluesky_pettingzoo.rendering.vertical_cr_renderer.haversine_distance",
        return_value=50.0,
    )
    @patch("bluesky_pettingzoo.rendering.common.pygame")
    @patch("bluesky_pettingzoo.rendering.vertical_cr_renderer.pygame")
    @patch("bluesky_pettingzoo.rendering.base_renderer.pygame")
    def test_render_frame_draws_aircraft_dots(
        self,
        mock_base_pygame,
        mock_vcr_pygame,
        mock_common_pygame,
        mock_haversine,
    ):
        mock_screen = MagicMock()
        mock_base_pygame.display.set_mode.return_value = mock_screen
        mock_base_pygame.Surface = MagicMock
        mock_base_pygame.font.Font.return_value = MagicMock()
        mock_base_pygame.time.Clock.return_value = MagicMock()
        mock_common_pygame.draw = MagicMock()

        from bluesky_pettingzoo.rendering.vertical_cr_renderer import VerticalCRRenderer

        renderer = VerticalCRRenderer(width=800, height=600)
        renderer.display()

        states = {
            "AC000": MagicMock(lat=40.0, lon=117.0, hdg=90.0, alt=35000.0),
            "AC001": MagicMock(lat=40.1, lon=117.1, hdg=270.0, alt=36000.0),
        }
        waypoints = {
            "AC000": {"lat": 40.5, "lon": 117.5, "alt": 35000.0, "hdg": 90.0},
            "AC001": {"lat": 39.5, "lon": 116.5, "alt": 36000.0, "hdg": 270.0},
        }
        renderer.render_frame(states=states, waypoints=waypoints)
        # Should draw aircraft dots (rect) and protection zones (circles)
        assert mock_common_pygame.draw.rect.call_count >= 2
        renderer.close()

    @patch(
        "bluesky_pettingzoo.rendering.vertical_cr_renderer.haversine_distance",
        return_value=50.0,
    )
    @patch("bluesky_pettingzoo.rendering.common.pygame")
    @patch("bluesky_pettingzoo.rendering.vertical_cr_renderer.pygame")
    @patch("bluesky_pettingzoo.rendering.base_renderer.pygame")
    def test_render_frame_draws_altitude_labels(
        self,
        mock_base_pygame,
        mock_vcr_pygame,
        mock_common_pygame,
        mock_haversine,
    ):
        mock_screen = MagicMock()
        mock_base_pygame.display.set_mode.return_value = mock_screen
        mock_base_pygame.Surface = MagicMock
        mock_font = MagicMock()
        mock_base_pygame.font.Font.return_value = mock_font
        mock_base_pygame.time.Clock.return_value = MagicMock()
        mock_common_pygame.draw = MagicMock()

        from bluesky_pettingzoo.rendering.vertical_cr_renderer import VerticalCRRenderer

        renderer = VerticalCRRenderer(width=800, height=600)
        renderer.display()

        states = {"AC000": MagicMock(lat=40.0, lon=117.0, hdg=90.0, alt=35000.0)}
        waypoints = {"AC000": {"lat": 40.5, "lon": 117.5, "alt": 35000.0, "hdg": 90.0}}
        renderer.render_frame(states=states, waypoints=waypoints)
        # Should render altitude text labels
        assert mock_font.render.call_count >= 1
        renderer.close()

    @patch(
        "bluesky_pettingzoo.rendering.vertical_cr_renderer.haversine_distance",
        return_value=50.0,
    )
    @patch("bluesky_pettingzoo.rendering.common.pygame")
    @patch("bluesky_pettingzoo.rendering.vertical_cr_renderer.pygame")
    @patch("bluesky_pettingzoo.rendering.base_renderer.pygame")
    def test_render_frame_draws_heading_lines(
        self,
        mock_base_pygame,
        mock_vcr_pygame,
        mock_common_pygame,
        mock_haversine,
    ):
        mock_screen = MagicMock()
        mock_base_pygame.display.set_mode.return_value = mock_screen
        mock_base_pygame.Surface = MagicMock
        mock_base_pygame.font.Font.return_value = MagicMock()
        mock_base_pygame.time.Clock.return_value = MagicMock()
        mock_common_pygame.draw = MagicMock()

        from bluesky_pettingzoo.rendering.vertical_cr_renderer import VerticalCRRenderer

        renderer = VerticalCRRenderer(width=800, height=600)
        renderer.display()

        states = {
            "AC000": MagicMock(lat=40.0, lon=117.0, hdg=90.0, alt=35000.0),
            "AC001": MagicMock(lat=40.1, lon=117.1, hdg=270.0, alt=36000.0),
        }
        renderer.render_frame(states=states)
        # Should draw heading lines (one per aircraft)
        assert mock_common_pygame.draw.line.call_count >= 2
        renderer.close()
