"""Tests for SectorCRRenderer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestSectorCRRendererImport:
    """Verify SectorCRRenderer can be imported."""

    def test_import(self):
        from bluesky_pettingzoo.rendering.sector_cr_renderer import SectorCRRenderer

        assert SectorCRRenderer is not None

    def test_inherits_base(self):
        from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer
        from bluesky_pettingzoo.rendering.sector_cr_renderer import SectorCRRenderer

        assert issubclass(SectorCRRenderer, BaseRenderer)


class TestSectorCRRendererFrame:
    """Verify SectorCR renders sector polygon, aircraft dots, protection zones, and heading lines."""  # noqa: E501

    @patch("bluesky_pettingzoo.rendering.sector_cr_renderer.haversine_distance", return_value=50.0)
    @patch("bluesky_pettingzoo.rendering.common.pygame")
    @patch("bluesky_pettingzoo.rendering.base_renderer.pygame")
    def test_render_frame_draws_aircraft_and_sector(
        self,
        mock_base_pygame,
        mock_common_pygame,
        mock_haversine,
    ):
        mock_screen = MagicMock()
        mock_base_pygame.display.set_mode.return_value = mock_screen
        mock_base_pygame.Surface = MagicMock
        mock_base_pygame.font.Font.return_value = MagicMock()
        mock_base_pygame.time.Clock.return_value = MagicMock()
        mock_common_pygame.draw = MagicMock()

        from bluesky_pettingzoo.rendering.sector_cr_renderer import SectorCRRenderer

        renderer = SectorCRRenderer(width=800, height=600)
        renderer.display()

        states = {
            "AC000": MagicMock(lat=40.0, lon=117.0, hdg=90.0, alt=35000.0),
            "AC001": MagicMock(lat=40.1, lon=117.1, hdg=270.0, alt=35000.0),
        }
        waypoints = {
            "AC000": {"lat": 40.5, "lon": 117.5, "alt": 35000.0, "hdg": 90.0},
            "AC001": {"lat": 39.5, "lon": 116.5, "alt": 35000.0, "hdg": 270.0},
        }
        sector_vertices = [(40.0, 116.5), (40.5, 117.5), (40.0, 118.0), (39.5, 117.0)]
        renderer.render_frame(
            states=states,
            waypoints=waypoints,
            sector_vertices=sector_vertices,
        )
        # Should draw sector polygon + aircraft dots (rect) + protection circles
        assert mock_common_pygame.draw.polygon.call_count >= 1  # sector polygon
        assert mock_common_pygame.draw.rect.call_count >= 2  # aircraft dots
        assert mock_common_pygame.draw.circle.call_count >= 2  # protection circles
        renderer.close()

    @patch("bluesky_pettingzoo.rendering.sector_cr_renderer.haversine_distance", return_value=50.0)
    @patch("bluesky_pettingzoo.rendering.common.pygame")
    @patch("bluesky_pettingzoo.rendering.base_renderer.pygame")
    def test_render_frame_without_sector(
        self,
        mock_base_pygame,
        mock_common_pygame,
        mock_haversine,
    ):
        mock_screen = MagicMock()
        mock_base_pygame.display.set_mode.return_value = mock_screen
        mock_base_pygame.Surface = MagicMock
        mock_base_pygame.font.Font.return_value = MagicMock()
        mock_base_pygame.time.Clock.return_value = MagicMock()
        mock_common_pygame.draw = MagicMock()

        from bluesky_pettingzoo.rendering.sector_cr_renderer import SectorCRRenderer

        renderer = SectorCRRenderer(width=800, height=600)
        renderer.display()

        states = {"AC000": MagicMock(lat=40.0, lon=117.0, hdg=90.0, alt=35000.0)}
        renderer.render_frame(states=states)
        # Should draw aircraft dot but no sector polygon
        assert mock_common_pygame.draw.rect.call_count >= 1
        renderer.close()

    @patch("bluesky_pettingzoo.rendering.sector_cr_renderer.haversine_distance", return_value=50.0)
    @patch("bluesky_pettingzoo.rendering.common.pygame")
    @patch("bluesky_pettingzoo.rendering.base_renderer.pygame")
    def test_render_frame_draws_heading_lines(
        self,
        mock_base_pygame,
        mock_common_pygame,
        mock_haversine,
    ):
        mock_screen = MagicMock()
        mock_base_pygame.display.set_mode.return_value = mock_screen
        mock_base_pygame.Surface = MagicMock
        mock_base_pygame.font.Font.return_value = MagicMock()
        mock_base_pygame.time.Clock.return_value = MagicMock()
        mock_common_pygame.draw = MagicMock()

        from bluesky_pettingzoo.rendering.sector_cr_renderer import SectorCRRenderer

        renderer = SectorCRRenderer(width=800, height=600)
        renderer.display()

        states = {
            "AC000": MagicMock(lat=40.0, lon=117.0, hdg=90.0, alt=35000.0),
            "AC001": MagicMock(lat=40.1, lon=117.1, hdg=270.0, alt=35000.0),
        }
        renderer.render_frame(states=states)
        # Should draw heading lines (one per aircraft)
        assert mock_common_pygame.draw.line.call_count >= 2
        renderer.close()
