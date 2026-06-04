"""Tests for assign_sector geometry utility."""

from __future__ import annotations

from bluesky_pettingzoo.utils.geometry import assign_sector

# Rectangular sectors (bounds format: [[lat_min, lon_min], [lat_max, lon_max]])
SECTORS_BOUNDS = [
    {"id": "sector_a", "bounds": [[39.0, 116.0], [39.5, 116.5]]},
    {"id": "sector_b", "bounds": [[39.0, 116.5], [39.5, 117.0]]},
]

# Polygon sectors
SECTORS_POLYGON = [
    {
        "id": "tri_a",
        "polygon": [(39.0, 116.0), (39.5, 116.0), (39.25, 116.5)],
    },
    {
        "id": "tri_b",
        "polygon": [(39.0, 116.5), (39.5, 116.5), (39.25, 117.0)],
    },
]


class TestAssignSectorBounds:
    """Test assign_sector with rectangular bounds."""

    def test_point_in_sector_a(self) -> None:
        """Point inside sector_a should return 'sector_a'."""
        result = assign_sector(39.25, 116.25, SECTORS_BOUNDS)
        assert result == "sector_a"

    def test_point_in_sector_b(self) -> None:
        """Point inside sector_b should return 'sector_b'."""
        result = assign_sector(39.25, 116.75, SECTORS_BOUNDS)
        assert result == "sector_b"

    def test_point_outside_all_sectors(self) -> None:
        """Point outside all sectors should return None."""
        result = assign_sector(40.0, 116.25, SECTORS_BOUNDS)
        assert result is None

    def test_point_on_boundary(self) -> None:
        """Point on shared boundary should return one of the sectors."""
        result = assign_sector(39.25, 116.5, SECTORS_BOUNDS)
        assert result in ("sector_a", "sector_b")

    def test_point_at_corner(self) -> None:
        """Point at corner of a sector should be inside."""
        result = assign_sector(39.0, 116.0, SECTORS_BOUNDS)
        assert result == "sector_a"

    def test_empty_sectors(self) -> None:
        """Empty sector list should return None."""
        result = assign_sector(39.25, 116.25, [])
        assert result is None


class TestAssignSectorPolygon:
    """Test assign_sector with polygon sectors."""

    def test_point_in_triangle_a(self) -> None:
        """Point inside triangle_a should return 'tri_a'."""
        # Centroid of triangle_a: (39.25, 116.167)
        result = assign_sector(39.25, 116.167, SECTORS_POLYGON)
        assert result == "tri_a"

    def test_point_in_triangle_b(self) -> None:
        """Point inside triangle_b should return 'tri_b'."""
        result = assign_sector(39.25, 116.833, SECTORS_POLYGON)
        assert result == "tri_b"

    def test_point_outside_polygons(self) -> None:
        """Point outside all polygons should return None."""
        result = assign_sector(39.8, 116.25, SECTORS_POLYGON)
        assert result is None


class TestAssignSectorMixed:
    """Test assign_sector with mixed bounds/polygon sectors."""

    def test_mixed_format(self) -> None:
        """Sectors can have either 'bounds' or 'polygon' key."""
        mixed = [
            {"id": "rect", "bounds": [[39.0, 116.0], [39.5, 116.5]]},
            {"id": "poly", "polygon": [(39.0, 116.5), (39.5, 116.5), (39.25, 117.0)]},
        ]
        assert assign_sector(39.25, 116.25, mixed) == "rect"
        assert assign_sector(39.25, 116.833, mixed) == "poly"
