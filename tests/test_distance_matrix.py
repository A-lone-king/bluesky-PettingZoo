"""Tests for batch distance matrix computation."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_pettingzoo.utils.geometry import haversine_distance, haversine_distance_matrix


class TestHaversineDistanceMatrix:
    """Test haversine_distance_matrix() against scalar haversine_distance()."""

    def test_single_point_returns_zero(self) -> None:
        """Single point distance matrix is 0."""
        lats = np.array([35.0])
        lons = np.array([-120.0])
        result = haversine_distance_matrix(lats, lons)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_two_points_symmetric(self) -> None:
        """Distance matrix is symmetric."""
        lats = np.array([35.0, 36.0])
        lons = np.array([-120.0, -121.0])
        result = haversine_distance_matrix(lats, lons)
        assert result.shape == (2, 2)
        assert result[0, 1] == pytest.approx(result[1, 0], rel=1e-10)

    def test_diagonal_is_zero(self) -> None:
        """Diagonal elements are zero."""
        lats = np.array([35.0, 36.0, 37.0])
        lons = np.array([-120.0, -121.0, -122.0])
        result = haversine_distance_matrix(lats, lons)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-6)

    def test_matches_scalar_version(self) -> None:
        """Matrix computation matches scalar haversine_distance for all pairs."""
        lats = np.array([35.0, 36.5, 40.0, 38.0])
        lons = np.array([-120.0, -121.5, -118.0, -119.0])
        result = haversine_distance_matrix(lats, lons)

        n = len(lats)
        for i in range(n):
            for j in range(n):
                expected = haversine_distance(lats[i], lons[i], lats[j], lons[j])
                assert result[i, j] == pytest.approx(expected, rel=1e-3)

    def test_known_distance(self) -> None:
        """Verify a known distance (roughly)."""
        # London (51.5, -0.1) to Paris (48.9, 2.3) ≈ 190-210 NM
        lats = np.array([51.5, 48.9])
        lons = np.array([-0.1, 2.3])
        result = haversine_distance_matrix(lats, lons)
        assert 180 < result[0, 1] < 220

    def test_output_dtype_float64(self) -> None:
        """Output is float64 numpy array."""
        lats = np.array([35.0, 36.0])
        lons = np.array([-120.0, -121.0])
        result = haversine_distance_matrix(lats, lons)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_five_points(self) -> None:
        """Matrix works for 5 points (typical scenario size)."""
        rng = np.random.default_rng(42)
        lats = rng.uniform(30, 45, size=5)
        lons = rng.uniform(-130, -110, size=5)
        result = haversine_distance_matrix(lats, lons)
        assert result.shape == (5, 5)
        # All off-diagonal should be positive
        for i in range(5):
            for j in range(5):
                if i != j:
                    assert result[i, j] > 0

    def test_twenty_points_performance(self) -> None:
        """Matrix for 20 points completes without error."""
        rng = np.random.default_rng(42)
        lats = rng.uniform(30, 45, size=20)
        lons = rng.uniform(-130, -110, size=20)
        result = haversine_distance_matrix(lats, lons)
        assert result.shape == (20, 20)
