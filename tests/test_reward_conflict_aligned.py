"""Tests for conflict penalty aligned with bluesky-gym (spec4 F1).

Verify that nmac_penalty=-1.0, warning_penalty=0.0, separation_penalty=0.0.
"""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


def make_state(lat: float = 39.25, lon: float = 116.25, alt: float = 35000.0) -> AircraftState:
    return AircraftState(
        id="AC001", lat=lat, lon=lon, alt=alt,
        hdg=90.0, tas=450.0, vs=0.0,
    )


def make_action() -> DiscreteAction:
    return DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)


@pytest.fixture
def config() -> dict:
    return {
        "components": {
            "conflict": {
                "enabled": True,
                "weight": 0.5,
                "nmac_penalty": -1.0,
                "warning_penalty": 0.0,
                "separation_penalty": 0.0,
                "thresholds": {
                    "nmac_horizontal_nm": 5,
                    "nmac_vertical_ft": 1000,
                    "warning_horizontal_nm": 10,
                    "warning_vertical_ft": 2000,
                }
            }
        }
    }


class TestNMACPenalty:
    """NMAC penalty should be -1.0 (aligns with bluesky-gym -1 * intruders)."""

    def test_nmac_penalty_value(self, config: dict) -> None:
        comp = ConflictPenalty(config)
        own = make_state(lat=39.25, lon=116.25, alt=35000.0)
        # Within NMAC threshold: <5nm horizontal, <1000ft vertical
        intruder = AircraftState(
            id="AC002", lat=39.25, lon=116.26, alt=35000.0,
            hdg=90.0, tas=450.0, vs=0.0,
        )
        action = make_action()

        result = comp.compute("AC001", own, action, own, {"AC001": own, "AC002": intruder})

        assert result == pytest.approx(-1.0)

    def test_nmac_penalty_multiple_intruders(self, config: dict) -> None:
        """Multiple NMAC intruders should return -1.0 (worst penalty, not sum)."""
        comp = ConflictPenalty(config)
        own = make_state(lat=39.25, lon=116.25, alt=35000.0)
        intruder1 = AircraftState(
            id="AC002", lat=39.25, lon=116.26, alt=35000.0,
            hdg=90.0, tas=450.0, vs=0.0,
        )
        intruder2 = AircraftState(
            id="AC003", lat=39.26, lon=116.25, alt=35000.0,
            hdg=90.0, tas=450.0, vs=0.0,
        )
        action = make_action()

        result = comp.compute(
            "AC001", own, action, own,
            {"AC001": own, "AC002": intruder1, "AC003": intruder2},
        )

        assert result == pytest.approx(-1.0)


class TestWarningPenalty:
    """Warning penalty should be 0.0 (disabled)."""

    def test_warning_penalty_zero(self, config: dict) -> None:
        comp = ConflictPenalty(config)
        own = make_state(lat=39.25, lon=116.25, alt=35000.0)
        # Within warning but outside NMAC: 5-10nm horizontal, <2000ft vertical
        other = AircraftState(
            id="AC002", lat=39.25, lon=116.40, alt=35000.0,
            hdg=90.0, tas=450.0, vs=0.0,
        )
        action = make_action()

        result = comp.compute("AC001", own, action, own, {"AC001": own, "AC002": other})

        assert result == pytest.approx(0.0)


class TestSeparationPenalty:
    """Separation penalty should be 0.0 (disabled)."""

    def test_separation_penalty_zero(self, config: dict) -> None:
        comp = ConflictPenalty(config)
        own = make_state(lat=39.25, lon=116.25, alt=35000.0)
        # Within NMAC horizontal but large vertical separation
        other = AircraftState(
            id="AC002", lat=39.25, lon=116.26, alt=38000.0,
            hdg=90.0, tas=450.0, vs=0.0,
        )
        action = make_action()

        result = comp.compute("AC001", own, action, own, {"AC001": own, "AC002": other})

        assert result == pytest.approx(0.0)


class TestNoConflict:
    """No penalty when aircraft are far apart."""

    def test_no_conflict_returns_zero(self, config: dict) -> None:
        comp = ConflictPenalty(config)
        own = make_state(lat=39.25, lon=116.25, alt=35000.0)
        # Far away: >10nm horizontal, >2000ft vertical
        other = AircraftState(
            id="AC002", lat=39.50, lon=116.50, alt=38000.0,
            hdg=90.0, tas=450.0, vs=0.0,
        )
        action = make_action()

        result = comp.compute("AC001", own, action, own, {"AC001": own, "AC002": other})

        assert result == pytest.approx(0.0)
