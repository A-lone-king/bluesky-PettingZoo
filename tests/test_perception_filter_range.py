"""Tests for perception filter range parameters (spec4 F1).

Verify that perception radius 20NM and altitude difference 3000ft filter correctly.
"""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.observations.filters import PerceptionFilter
from bluesky_pettingzoo.utils.types import AircraftState


def make_state(lat: float = 39.25, lon: float = 116.25, alt: float = 35000.0) -> AircraftState:
    return AircraftState(
        id="AC001",
        lat=lat,
        lon=lon,
        alt=alt,
        hdg=90.0,
        tas=450.0,
        vs=0.0,
    )


@pytest.fixture
def config() -> dict:
    return {
        "observation": {
            "perception_radius_nm": 20,
            "perception_alt_diff_ft": 3000,
            "max_observable_aircraft": 10,
        }
    }


class TestPerceptionRadius:
    """Aircraft outside 20NM radius should be filtered out."""

    def test_within_radius_visible(self, config: dict) -> None:
        """Aircraft within 20NM should be visible."""
        filt = PerceptionFilter(config)
        own = make_state(lat=39.25, lon=116.25)
        # ~10NM away
        other = make_state(lat=39.25, lon=116.45, alt=35000.0)
        other = AircraftState(
            id="AC002", lat=39.25, lon=116.45, alt=35000.0, hdg=90.0, tas=450.0, vs=0.0
        )

        results = filt.filter(own, [other])

        assert len(results) == 1
        assert results[0]["state"].id == "AC002"

    def test_outside_radius_filtered(self, config: dict) -> None:
        """Aircraft outside 20NM should be filtered out."""
        filt = PerceptionFilter(config)
        own = make_state(lat=39.25, lon=116.25)
        # ~30NM away
        other = AircraftState(
            id="AC002", lat=39.25, lon=116.85, alt=35000.0, hdg=90.0, tas=450.0, vs=0.0
        )

        results = filt.filter(own, [other])

        assert len(results) == 0

    def test_at_radius_boundary_visible(self, config: dict) -> None:
        """Aircraft at exactly 20NM should be visible (with tolerance)."""
        filt = PerceptionFilter(config)
        own = make_state(lat=39.25, lon=116.25)
        # ~20NM away (with tolerance)
        other = AircraftState(
            id="AC002", lat=39.25, lon=116.65, alt=35000.0, hdg=90.0, tas=450.0, vs=0.0
        )

        results = filt.filter(own, [other])

        # Should be visible due to 0.05nm tolerance
        assert len(results) == 1


class TestAltitudeDifference:
    """Aircraft with altitude difference > 3000ft should be filtered out."""

    def test_within_alt_visible(self, config: dict) -> None:
        """Aircraft within 3000ft altitude difference should be visible."""
        filt = PerceptionFilter(config)
        own = make_state(alt=35000.0)
        other = AircraftState(
            id="AC002", lat=39.25, lon=116.25, alt=34000.0, hdg=90.0, tas=450.0, vs=0.0
        )

        results = filt.filter(own, [other])

        assert len(results) == 1

    def test_outside_alt_filtered(self, config: dict) -> None:
        """Aircraft with > 3000ft altitude difference should be filtered out."""
        filt = PerceptionFilter(config)
        own = make_state(alt=35000.0)
        other = AircraftState(
            id="AC002", lat=39.25, lon=116.25, alt=31000.0, hdg=90.0, tas=450.0, vs=0.0
        )

        results = filt.filter(own, [other])

        assert len(results) == 0

    def test_at_alt_boundary_visible(self, config: dict) -> None:
        """Aircraft at exactly 3000ft difference should be visible."""
        filt = PerceptionFilter(config)
        own = make_state(alt=35000.0)
        other = AircraftState(
            id="AC002", lat=39.25, lon=116.25, alt=32000.0, hdg=90.0, tas=450.0, vs=0.0
        )

        results = filt.filter(own, [other])

        assert len(results) == 1


class TestMaxObservable:
    """Results should be truncated to max_observable."""

    def test_truncates_to_max(self, config: dict) -> None:
        """Results should be truncated to max_observable."""
        config["observation"]["max_observable_aircraft"] = 2
        filt = PerceptionFilter(config)
        own = make_state(alt=35000.0)

        others = [
            AircraftState(
                id=f"AC{i:03d}",
                lat=39.25,
                lon=116.25 + i * 0.01,
                alt=35000.0,
                hdg=90.0,
                tas=450.0,
                vs=0.0,
            )
            for i in range(5)
        ]

        results = filt.filter(own, others)

        assert len(results) == 2


class TestSortedByDistance:
    """Results should be sorted by distance ascending."""

    def test_sorted_ascending(self, config: dict) -> None:
        """Results should be sorted by distance (closest first)."""
        filt = PerceptionFilter(config)
        own = make_state(lat=39.25, lon=116.25)

        # Create aircraft at different distances
        far = AircraftState(
            id="AC_FAR", lat=39.25, lon=116.55, alt=35000.0, hdg=90.0, tas=450.0, vs=0.0
        )
        near = AircraftState(
            id="AC_NEAR", lat=39.25, lon=116.35, alt=35000.0, hdg=90.0, tas=450.0, vs=0.0
        )

        results = filt.filter(own, [far, near])

        assert results[0]["state"].id == "AC_NEAR"
        assert results[1]["state"].id == "AC_FAR"
