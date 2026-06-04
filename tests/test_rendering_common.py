"""Tests for shared rendering utility functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from bluesky_pettingzoo.rendering.common import (
    compute_pixels_per_nm,
    draw_aircraft,
    draw_aircraft_dot,
    draw_heading_line,
    draw_nmac_circle,
    draw_sector_polygon,
    draw_waypoint,
    latlon_to_pixel,
)


class TestLatlonToPixel:
    """Verify coordinate conversion from lat/lon to pixel."""

    def test_returns_tuple(self):
        result = latlon_to_pixel(
            lat=40.0,
            lon=117.0,
            bounds={"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0},
            width=800,
            height=600,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_center_maps_to_center(self):
        result = latlon_to_pixel(
            lat=40.0,
            lon=117.0,
            bounds={"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0},
            width=800,
            height=600,
        )
        assert result == (400, 300)

    def test_min_corner(self):
        result = latlon_to_pixel(
            lat=39.0,
            lon=116.0,
            bounds={"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0},
            width=800,
            height=600,
        )
        # lat_min is at bottom (pixel y = height), lon_min is at left (pixel x = 0)
        assert result[0] == 0
        assert result[1] == 600

    def test_max_corner(self):
        result = latlon_to_pixel(
            lat=41.0,
            lon=118.0,
            bounds={"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0},
            width=800,
            height=600,
        )
        assert result[0] == 800
        assert result[1] == 0


class TestDrawAircraft:
    """Verify draw_aircraft calls Pygame correctly."""

    @patch("bluesky_pettingzoo.rendering.common.pygame")
    def test_draws_triangle(self, mock_pygame):
        mock_screen = MagicMock()
        draw_aircraft(
            screen=mock_screen,
            x=400,
            y=300,
            heading=90.0,
            color=(0, 255, 0),
            size=10,
        )
        mock_pygame.draw.polygon.assert_called_once()

    @patch("bluesky_pettingzoo.rendering.common.pygame")
    def test_draws_with_different_colors(self, mock_pygame):
        mock_screen = MagicMock()
        draw_aircraft(mock_screen, 100, 200, 0.0, color=(255, 0, 0))
        call_args = mock_pygame.draw.polygon.call_args
        # pygame.draw.polygon(screen, color, points) — color is 2nd positional arg
        assert call_args[0][1] == (255, 0, 0)


class TestDrawWaypoint:
    """Verify draw_waypoint calls Pygame correctly."""

    @patch("bluesky_pettingzoo.rendering.common.pygame")
    def test_draws_circle(self, mock_pygame):
        mock_screen = MagicMock()
        draw_waypoint(mock_screen, 400, 300, color=(255, 255, 0))
        mock_pygame.draw.circle.assert_called_once()


class TestDrawNMACCircle:
    """Verify draw_nmac_circle calls Pygame correctly."""

    @patch("bluesky_pettingzoo.rendering.common.pygame")
    def test_draws_circle(self, mock_pygame):
        mock_screen = MagicMock()
        draw_nmac_circle(mock_screen, 400, 300, radius_nm=5.0, pixels_per_nm=10.0)
        mock_pygame.draw.circle.assert_called_once()


class TestDrawSectorPolygon:
    """Verify draw_sector_polygon calls Pygame correctly."""

    @patch("bluesky_pettingzoo.rendering.common.pygame")
    def test_draws_polygon(self, mock_pygame):
        mock_screen = MagicMock()
        vertices = [(40.0, 117.0), (40.5, 117.5), (40.0, 118.0)]
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        draw_sector_polygon(mock_screen, vertices, bounds, width=800, height=600)
        mock_pygame.draw.polygon.assert_called_once()


class TestDrawAircraftDot:
    """Verify draw_aircraft_dot draws a small filled rectangle."""

    @patch("bluesky_pettingzoo.rendering.common.pygame")
    def test_draws_rect(self, mock_pygame):
        mock_screen = MagicMock()
        draw_aircraft_dot(mock_screen, 400, 300, color=(0, 0, 0), size=4)
        mock_pygame.draw.rect.assert_called_once()

    @patch("bluesky_pettingzoo.rendering.common.pygame")
    def test_default_color_is_black(self, mock_pygame):
        mock_screen = MagicMock()
        draw_aircraft_dot(mock_screen, 400, 300)
        call_args = mock_pygame.draw.rect.call_args
        assert call_args[0][1] == (0, 0, 0)

    @patch("bluesky_pettingzoo.rendering.common.pygame")
    def test_noop_when_pygame_none(self, mock_pygame):
        import bluesky_pettingzoo.rendering.common as mod

        original = mod.pygame
        mod.pygame = None
        try:
            draw_aircraft_dot(MagicMock(), 400, 300)
        finally:
            mod.pygame = original


class TestDrawHeadingLine:
    """Verify draw_heading_line draws a line from aircraft position."""

    @patch("bluesky_pettingzoo.rendering.common.pygame")
    def test_draws_line(self, mock_pygame):
        mock_screen = MagicMock()
        draw_heading_line(mock_screen, 400, 300, heading=90.0, length=30)
        mock_pygame.draw.line.assert_called_once()

    @patch("bluesky_pettingzoo.rendering.common.pygame")
    def test_north_heading_goes_up(self, mock_pygame):
        mock_screen = MagicMock()
        draw_heading_line(mock_screen, 400, 300, heading=0.0, length=30)
        call_args = mock_pygame.draw.line.call_args
        # heading=0 (north) means line goes upward (y decreases)
        end_y = call_args[0][3][1]
        assert end_y < 300

    @patch("bluesky_pettingzoo.rendering.common.pygame")
    def test_south_heading_goes_down(self, mock_pygame):
        mock_screen = MagicMock()
        draw_heading_line(mock_screen, 400, 300, heading=180.0, length=30)
        call_args = mock_pygame.draw.line.call_args
        end_y = call_args[0][3][1]
        assert end_y > 300

    @patch("bluesky_pettingzoo.rendering.common.pygame")
    def test_handles_dict_heading(self, mock_pygame):
        mock_screen = MagicMock()
        draw_heading_line(mock_screen, 400, 300, heading={"hdg": 90.0}, length=30)
        mock_pygame.draw.line.assert_called_once()

    @patch("bluesky_pettingzoo.rendering.common.pygame")
    def test_noop_when_pygame_none(self, mock_pygame):
        import bluesky_pettingzoo.rendering.common as mod

        original = mod.pygame
        mod.pygame = None
        try:
            draw_heading_line(MagicMock(), 400, 300, heading=90.0)
        finally:
            mod.pygame = original


class TestComputePixelsPerNm:
    """Verify pixel-per-nautical-mile computation."""

    def test_basic_computation(self):
        # 2 degrees lat = ~120 NM, height=600 => 5 pixels/NM
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        result = compute_pixels_per_nm(bounds, height=600)
        assert abs(result - 5.0) < 0.01

    def test_different_height(self):
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        result = compute_pixels_per_nm(bounds, height=1200)
        assert abs(result - 10.0) < 0.01

    def test_zero_range_returns_default(self):
        bounds = {"lat_min": 40.0, "lat_max": 40.0, "lon_min": 117.0, "lon_max": 117.0}
        result = compute_pixels_per_nm(bounds, height=600)
        assert result == 10.0  # default fallback
