"""Tests for perception range filter module."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_pettingzoo.observations.filters import PerceptionFilter
from bluesky_pettingzoo.utils.types import AircraftState


def make_state(
    acid: str,
    lat: float,
    lon: float,
    alt: float,
    hdg: float = 90.0,
    tas: float = 450.0,
    vs: float = 0.0,
) -> AircraftState:
    """Helper to create aircraft state."""
    return AircraftState(
        id=acid,
        lat=lat,
        lon=lon,
        alt=alt,
        hdg=hdg,
        tas=tas,
        vs=vs,
    )


class TestFilterNoAircraft:
    """Test filtering with no other aircraft."""

    def test_filter_no_aircraft(self, default_config: dict) -> None:
        """No other aircraft should return empty list."""
        filt = PerceptionFilter(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        others: list[AircraftState] = []

        result = filt.filter(own, others)

        assert result == []


class TestFilterHorizontalRadius:
    """Test horizontal radius filtering."""

    def test_filter_within_radius(self, default_config: dict) -> None:
        """Aircraft at 10NM (< 20NM) should be observable."""
        filt = PerceptionFilter(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        other = make_state("AC001", 39.33, 116.25, 35000.0)  # ~10NM north

        result = filt.filter(own, [other])

        assert len(result) == 1
        assert result[0]["state"].id == "AC001"

    def test_filter_outside_radius(self, default_config: dict) -> None:
        """Aircraft at 30NM (> 20NM) should not be observable."""
        filt = PerceptionFilter(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        other = make_state("AC001", 39.75, 116.25, 35000.0)  # ~30NM north

        result = filt.filter(own, [other])

        assert len(result) == 0

    def test_filter_at_boundary(self, default_config: dict) -> None:
        """Aircraft at exactly 20NM should be observable (inclusive)."""
        filt = PerceptionFilter(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        # ~20NM north: 20NM / 60 ≈ 0.3333 degrees
        other = make_state("AC001", 39.5833, 116.25, 35000.0)

        result = filt.filter(own, [other])

        # Should be included (boundary is inclusive)
        assert len(result) == 1


class TestFilterVerticalRange:
    """Test vertical range filtering."""

    def test_filter_within_alt_range(self, default_config: dict) -> None:
        """Aircraft with 2000ft difference (< 3000ft) should be observable."""
        filt = PerceptionFilter(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        other = make_state("AC001", 39.2501, 116.25, 33000.0)  # Very close horizontally

        result = filt.filter(own, [other])

        assert len(result) == 1

    def test_filter_outside_alt_range(self, default_config: dict) -> None:
        """Aircraft with 5000ft difference (> 3000ft) should not be observable."""
        filt = PerceptionFilter(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        other = make_state("AC001", 39.2501, 116.25, 30000.0)  # Very close horizontally

        result = filt.filter(own, [other])

        assert len(result) == 0

    def test_filter_at_alt_boundary(self, default_config: dict) -> None:
        """Aircraft with exactly 3000ft difference should be observable (inclusive)."""
        filt = PerceptionFilter(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        other = make_state("AC001", 39.2501, 116.25, 32000.0)  # Very close horizontally

        result = filt.filter(own, [other])

        assert len(result) == 1


class TestFilterCombined:
    """Test combined horizontal and vertical filtering."""

    def test_filter_combined(self, default_config: dict) -> None:
        """Aircraft within horizontal range but outside vertical range should not be observable."""
        filt = PerceptionFilter(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        # 10NM away horizontally, but 5000ft difference vertically
        other = make_state("AC001", 39.33, 116.25, 30000.0)

        result = filt.filter(own, [other])

        assert len(result) == 0

    def test_filter_both_in_range(self, default_config: dict) -> None:
        """Aircraft within both horizontal and vertical range should be observable."""
        filt = PerceptionFilter(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        # 10NM away horizontally, 2000ft difference vertically
        other = make_state("AC001", 39.33, 116.25, 33000.0)

        result = filt.filter(own, [other])

        assert len(result) == 1


class TestFilterMaxObservable:
    """Test maximum observable aircraft limit."""

    def test_filter_max_observable(self, default_config: dict) -> None:
        """Should truncate to max observable aircraft count."""
        filt = PerceptionFilter(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)

        # Create 15 aircraft within range (max is 10)
        others = []
        for i in range(15):
            lat_offset = (i + 1) * 0.01  # Each ~0.6NM apart
            others.append(make_state(f"AC{i:03d}", 39.25 + lat_offset, 116.25, 35000.0))

        result = filt.filter(own, others)

        assert len(result) == 10  # MAX_OBSERVABLE_AIRCRAFT


class TestFilterSortedByDistance:
    """Test sorting by distance."""

    def test_filter_sorted_by_distance(self, default_config: dict) -> None:
        """Results should be sorted by distance (ascending)."""
        filt = PerceptionFilter(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)

        # Create aircraft at different distances
        near = make_state("NEAR", 39.26, 116.25, 35000.0)    # ~0.6NM
        mid = make_state("MID", 39.30, 116.25, 35000.0)      # ~3NM
        far = make_state("FAR", 39.40, 116.25, 35000.0)      # ~9NM

        result = filt.filter(own, [far, near, mid])

        assert len(result) == 3
        assert result[0]["state"].id == "NEAR"
        assert result[1]["state"].id == "MID"
        assert result[2]["state"].id == "FAR"

    def test_filter_result_contains_distance(self, default_config: dict) -> None:
        """Each result should contain distance and bearing information."""
        filt = PerceptionFilter(default_config)
        own = make_state("OWN", 39.25, 116.25, 35000.0)
        other = make_state("AC001", 39.33, 116.25, 35000.0)

        result = filt.filter(own, [other])

        assert len(result) == 1
        assert "distance_nm" in result[0]
        assert "bearing_deg" in result[0]
        assert "state" in result[0]
        assert result[0]["distance_nm"] > 0
        assert 0 <= result[0]["bearing_deg"] <= 360
