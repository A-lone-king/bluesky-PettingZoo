"""Tests for shared rendering utility functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bluesky_pettingzoo.rendering.common import (
    latlon_to_pixel,
    draw_aircraft,
    draw_waypoint,
    draw_nmac_circle,
    draw_sector_polygon,
)


class TestLatlonToPixel:
    """Verify coordinate conversion from lat/lon to pixel."""

    def test_returns_tuple(self):
        result = latlon_to_pixel(
            lat=40.0, lon=117.0,
            bounds={"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0},
            width=800, height=600,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_center_maps_to_center(self):
        result = latlon_to_pixel(
            lat=40.0, lon=117.0,
            bounds={"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0},
            width=800, height=600,
        )
        assert result == (400, 300)

    def test_min_corner(self):
        result = latlon_to_pixel(
            lat=39.0, lon=116.0,
            bounds={"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0},
            width=800, height=600,
        )
        # lat_min is at bottom (pixel y = height), lon_min is at left (pixel x = 0)
        assert result[0] == 0
        assert result[1] == 600

    def test_max_corner(self):
        result = latlon_to_pixel(
            lat=41.0, lon=118.0,
            bounds={"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0},
            width=800, height=600,
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
            x=400, y=300, heading=90.0,
            color=(0, 255, 0), size=10,
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
