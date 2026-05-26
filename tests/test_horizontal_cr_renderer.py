"""Tests for HorizontalCRRenderer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestHorizontalCRRendererImport:
    """Verify HorizontalCRRenderer can be imported."""

    def test_import(self):
        from bluesky_pettingzoo.rendering.horizontal_cr_renderer import HorizontalCRRenderer
        assert HorizontalCRRenderer is not None

    def test_inherits_base(self):
        from bluesky_pettingzoo.rendering.horizontal_cr_renderer import HorizontalCRRenderer
        from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer
        assert issubclass(HorizontalCRRenderer, BaseRenderer)


class TestHorizontalCRRendererFrame:
    """Verify HorizontalCR renders aircraft, waypoints, and NMAC circles."""

    @patch("bluesky_pettingzoo.rendering.common.pygame")
    @patch("bluesky_pettingzoo.rendering.horizontal_cr_renderer.pygame")
    @patch("bluesky_pettingzoo.rendering.base_renderer.pygame")
    def test_render_frame_draws_aircraft(self, mock_base_pygame, mock_hcr_pygame, mock_common_pygame):
        mock_screen = MagicMock()
        mock_base_pygame.display.set_mode.return_value = mock_screen
        mock_base_pygame.Surface = MagicMock
        mock_base_pygame.font.Font.return_value = MagicMock()
        mock_base_pygame.time.Clock.return_value = MagicMock()
        mock_common_pygame.draw = MagicMock()

        from bluesky_pettingzoo.rendering.horizontal_cr_renderer import HorizontalCRRenderer
        renderer = HorizontalCRRenderer(width=800, height=600)
        renderer.display()

        states = {
            "AC000": MagicMock(lat=40.0, lon=117.0, hdg=90.0, alt=35000.0),
            "AC001": MagicMock(lat=40.1, lon=117.1, hdg=270.0, alt=35000.0),
        }
        waypoints = {
            "AC000": {"lat": 40.5, "lon": 117.5, "alt": 35000.0, "hdg": 90.0},
            "AC001": {"lat": 39.5, "lon": 116.5, "alt": 35000.0, "hdg": 270.0},
        }
        renderer.render_frame(states=states, waypoints=waypoints)
        # Should draw aircraft polygons
        assert mock_common_pygame.draw.polygon.call_count >= 2
        renderer.close()

    @patch("bluesky_pettingzoo.rendering.common.pygame")
    @patch("bluesky_pettingzoo.rendering.horizontal_cr_renderer.pygame")
    @patch("bluesky_pettingzoo.rendering.base_renderer.pygame")
    def test_render_frame_draws_waypoints(self, mock_base_pygame, mock_hcr_pygame, mock_common_pygame):
        mock_screen = MagicMock()
        mock_base_pygame.display.set_mode.return_value = mock_screen
        mock_base_pygame.Surface = MagicMock
        mock_base_pygame.font.Font.return_value = MagicMock()
        mock_base_pygame.time.Clock.return_value = MagicMock()
        mock_common_pygame.draw = MagicMock()

        from bluesky_pettingzoo.rendering.horizontal_cr_renderer import HorizontalCRRenderer
        renderer = HorizontalCRRenderer(width=800, height=600)
        renderer.display()

        states = {"AC000": MagicMock(lat=40.0, lon=117.0, hdg=90.0, alt=35000.0)}
        waypoints = {"AC000": {"lat": 40.5, "lon": 117.5, "alt": 35000.0, "hdg": 90.0}}
        renderer.render_frame(states=states, waypoints=waypoints)
        # Should draw waypoint circles
        assert mock_common_pygame.draw.circle.call_count >= 1
        renderer.close()
