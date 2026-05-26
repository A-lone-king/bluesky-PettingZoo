"""Geometric calculation utilities for ATM."""

from __future__ import annotations

import math

import numpy as np


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


def haversine_distance_matrix(
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Compute pairwise haversine distance matrix using numpy vectorization.

    Args:
        lats: Array of latitudes (degrees), shape (n,).
        lons: Array of longitudes (degrees), shape (n,).

    Returns:
        Distance matrix (n, n) in nautical miles.
    """
    R_nm = 3440.065

    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)

    # Broadcast: (n,1) vs (1,n)
    dlat = lats_rad[:, None] - lats_rad[None, :]
    dlon = lons_rad[:, None] - lons_rad[None, :]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lats_rad[:, None]) * np.cos(lats_rad[None, :]) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

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


def point_in_polygon(
    lat: float,
    lon: float,
    polygon: list[tuple[float, float]],
) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm.

    Args:
        lat: Point latitude.
        lon: Point longitude.
        polygon: List of (lat, lon) vertices.

    Returns:
        True if point is inside the polygon.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        if ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


def assign_sector(
    lat: float,
    lon: float,
    sectors: list[dict[str, object]],
) -> str | None:
    """Assign a point to the first matching sector.

    Each sector dict must have an ``id`` key and either:
    - ``bounds``: ``[[lat_min, lon_min], [lat_max, lon_max]]``
    - ``polygon``: list of ``(lat, lon)`` vertices

    Args:
        lat: Point latitude.
        lon: Point longitude.
        sectors: List of sector definitions.

    Returns:
        Sector id if point is inside a sector, otherwise ``None``.
    """
    for sector in sectors:
        sid = sector["id"]
        if "polygon" in sector:
            if point_in_polygon(lat, lon, sector["polygon"]):  # type: ignore[arg-type]
                return sid  # type: return-value
        elif "bounds" in sector:
            (lat_min, lon_min), (lat_max, lon_max) = sector["bounds"]  # type: ignore[misc]
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return sid
    return None


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    """2D cross product of vectors (ax,ay) and (bx,by)."""
    return ax * by - ay * bx


def segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> bool:
    """Check if two 2D line segments (p1-p2) and (p3-p4) intersect.

    Uses the orientation-based algorithm.  Segments sharing an endpoint
    are considered intersecting.

    Args:
        p1: First endpoint of segment 1 (x, y).
        p2: Second endpoint of segment 1 (x, y).
        p3: First endpoint of segment 2 (x, y).
        p4: Second endpoint of segment 2 (x, y).

    Returns:
        True if the segments intersect (including at endpoints).
    """

    def _orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return _cross(b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1])

    def _on_segment(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
        return (
            min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        )

    o1 = _orient(p1, p2, p3)
    o2 = _orient(p1, p2, p4)
    o3 = _orient(p3, p4, p1)
    o4 = _orient(p3, p4, p2)

    # General case
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True

    # Collinear special cases
    if abs(o1) < 1e-10 and _on_segment(p1, p2, p3):
        return True
    if abs(o2) < 1e-10 and _on_segment(p1, p2, p4):
        return True
    if abs(o3) < 1e-10 and _on_segment(p3, p4, p1):
        return True
    if abs(o4) < 1e-10 and _on_segment(p3, p4, p2):
        return True

    return False


def generate_polygon(
    rng: np.random.RandomState,
    center_lat: float,
    center_lon: float,
    num_vertices: int = 6,
    radius_deg: float = 0.15,
    min_area_deg2: float = 0.0,
) -> list[tuple[float, float]]:
    """Generate a random convex polygon around a center point.

    Args:
        rng: Random number generator.
        center_lat: Center latitude.
        center_lon: Center longitude.
        num_vertices: Number of polygon vertices.
        radius_deg: Approximate radius in degrees.
        min_area_deg2: Minimum area in square degrees (adds vertices if needed).

    Returns:
        List of (lat, lon) vertices in clockwise order.
    """
    angles = np.sort(rng.uniform(0, 2 * np.pi, num_vertices))
    radii = rng.uniform(radius_deg * 0.6, radius_deg, num_vertices)
    vertices = []
    for angle, r in zip(angles, radii):
        v_lat = center_lat + r * math.cos(angle)
        v_lon = center_lon + r * math.sin(angle)
        vertices.append((v_lat, v_lon))
    return vertices
