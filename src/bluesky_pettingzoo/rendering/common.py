"""Shared rendering utilities — coordinate conversion and drawing primitives."""

from __future__ import annotations

import math
from typing import Any

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]


def latlon_to_pixel(
    lat: float,
    lon: float,
    bounds: dict[str, float],
    width: int,
    height: int,
) -> tuple[int, int]:
    """Convert lat/lon coordinates to pixel position.

    Maps lat/lon within bounds to screen pixels. Latitude increases
    upward (y=0 at top), longitude increases rightward.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        bounds: Dict with lat_min, lat_max, lon_min, lon_max.
        width: Screen width in pixels.
        height: Screen height in pixels.

    Returns:
        (x, y) pixel coordinates.
    """
    lat_range = bounds["lat_max"] - bounds["lat_min"]
    lon_range = bounds["lon_max"] - bounds["lon_min"]
    if lat_range == 0 or lon_range == 0:
        return (width // 2, height // 2)

    x = int((lon - bounds["lon_min"]) / lon_range * width)
    y = int((bounds["lat_max"] - lat) / lat_range * height)
    return (x, y)


def draw_aircraft(
    screen: Any,
    x: int,
    y: int,
    heading: float,
    color: tuple[int, int, int] = (0, 255, 0),
    size: int = 10,
) -> None:
    """Draw an aircraft as a rotated triangle.

    Args:
        screen: Pygame surface.
        x: Pixel x coordinate.
        y: Pixel y coordinate.
        heading: Heading in degrees (0=north, 90=east).
        color: RGB color tuple.
        size: Triangle size in pixels.
    """
    if pygame is None:
        return
    hdg_rad = math.radians(heading)
    # Triangle points: nose, left wing, right wing
    nose = (
        x + int(size * math.sin(hdg_rad)),
        y - int(size * math.cos(hdg_rad)),
    )
    left = (
        x + int(size * 0.6 * math.sin(hdg_rad + 2.5)),
        y - int(size * 0.6 * math.cos(hdg_rad + 2.5)),
    )
    right = (
        x + int(size * 0.6 * math.sin(hdg_rad - 2.5)),
        y - int(size * 0.6 * math.cos(hdg_rad - 2.5)),
    )
    pygame.draw.polygon(screen, color, [nose, left, right])


def draw_waypoint(
    screen: Any,
    x: int,
    y: int,
    color: tuple[int, int, int] = (255, 255, 0),
    radius: int = 5,
) -> None:
    """Draw a waypoint as a small circle.

    Args:
        screen: Pygame surface.
        x: Pixel x coordinate.
        y: Pixel y coordinate.
        color: RGB color tuple.
        radius: Circle radius in pixels.
    """
    if pygame is None:
        return
    pygame.draw.circle(screen, color, (x, y), radius)


def draw_nmac_circle(
    screen: Any,
    x: int,
    y: int,
    radius_nm: float = 5.0,
    pixels_per_nm: float = 10.0,
    color: tuple[int, int, int] = (255, 0, 0),
    width: int = 1,
) -> None:
    """Draw a NMAC conflict circle around an aircraft.

    Args:
        screen: Pygame surface.
        x: Pixel x coordinate.
        y: Pixel y coordinate.
        radius_nm: NMAC radius in nautical miles.
        pixels_per_nm: Scale factor.
        color: RGB color tuple.
        width: Line width.
    """
    if pygame is None:
        return
    pixel_radius = int(radius_nm * pixels_per_nm)
    pygame.draw.circle(screen, color, (x, y), pixel_radius, width)


def draw_sector_polygon(
    screen: Any,
    vertices: list[tuple[float, float]],
    bounds: dict[str, float],
    width: int,
    height: int,
    color: tuple[int, int, int] = (100, 100, 255),
    line_width: int = 2,
) -> None:
    """Draw a sector polygon from lat/lon vertices.

    Args:
        screen: Pygame surface.
        vertices: List of (lat, lon) tuples.
        bounds: Airspace bounds dict.
        width: Screen width.
        height: Screen height.
        color: RGB color tuple.
        line_width: Line width.
    """
    if pygame is None or not vertices:
        return
    pixel_points = [
        latlon_to_pixel(lat, lon, bounds, width, height)
        for lat, lon in vertices
    ]
    pygame.draw.polygon(screen, color, pixel_points, line_width)
