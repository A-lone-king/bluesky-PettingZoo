"""Geometric calculation utilities for ATM."""

from __future__ import annotations

import math


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate the great-circle distance between two points in nautical miles.

    Uses the Haversine formula.

    Args:
        lat1: Latitude of point 1 (degrees)
        lon1: Longitude of point 1 (degrees)
        lat2: Latitude of point 2 (degrees)
        lon2: Longitude of point 2 (degrees)

    Returns:
        Distance in nautical miles
    """
    R_nm = 3440.065  # Earth radius in nautical miles

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R_nm * c


def bearing(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate the initial bearing from point 1 to point 2.

    Args:
        lat1: Latitude of point 1 (degrees)
        lon1: Longitude of point 1 (degrees)
        lat2: Latitude of point 2 (degrees)
        lon2: Longitude of point 2 (degrees)

    Returns:
        Bearing in degrees (0-360, where 0 is north)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)

    x = math.sin(dlon_rad) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
        lat2_rad
    ) * math.cos(dlon_rad)

    bearing_rad = math.atan2(x, y)
    bearing_deg = math.degrees(bearing_rad)

    return (bearing_deg + 360) % 360


def relative_position(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> tuple[float, float]:
    """Calculate relative position (distance and bearing) from point 1 to point 2.

    Args:
        lat1: Latitude of point 1 (degrees)
        lon1: Longitude of point 1 (degrees)
        lat2: Latitude of point 2 (degrees)
        lon2: Longitude of point 2 (degrees)

    Returns:
        Tuple of (distance_nm, bearing_deg)
    """
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    bear = bearing(lat1, lon1, lat2, lon2)
    return dist, bear


def point_at_distance(
    lat: float,
    lon: float,
    distance_nm: float,
    bearing_deg: float,
) -> tuple[float, float]:
    """Calculate a point at a given distance and bearing from origin.

    Args:
        lat: Origin latitude (degrees).
        lon: Origin longitude (degrees).
        distance_nm: Distance in nautical miles.
        bearing_deg: Bearing in degrees (0=north, 90=east).

    Returns:
        (lat, lon) of the destination point in degrees.
    """
    R_nm = 3440.065  # Earth radius in nautical miles
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    brng = math.radians(bearing_deg)
    d = distance_nm / R_nm

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d)
        + math.cos(lat1) * math.sin(d) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def project_position(
    lat: float,
    lon: float,
    hdg_deg: float,
    tas_kt: float,
    dt_s: float,
) -> tuple[float, float]:
    """Project aircraft position forward by dt_s seconds using heading and speed.

    Args:
        lat: Current latitude in degrees.
        lon: Current longitude in degrees.
        hdg_deg: Heading in degrees (true north clockwise).
        tas_kt: True airspeed in knots.
        dt_s: Time step in seconds.

    Returns:
        (new_lat, new_lon) in degrees.
    """
    hdg_rad = math.radians(hdg_deg)
    dist_nm = tas_kt * dt_s / 3600.0
    dlat = math.cos(hdg_rad) * dist_nm / 60.0
    dlon = math.sin(hdg_rad) * dist_nm / (60.0 * max(math.cos(math.radians(lat)), 1e-6))
    return lat + dlat, lon + dlon
