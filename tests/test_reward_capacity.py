"""Tests for capacity violation penalty reward component (T-V14)."""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.rewards.components.capacity import CapacityPenalty
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


def make_state(
    acid: str,
    lat: float = 39.25,
    lon: float = 116.25,
    alt: float = 35000.0,
    hdg: float = 90.0,
    tas: float = 450.0,
    vs: float = 0.0,
) -> AircraftState:
    return AircraftState(
        id=acid, lat=lat, lon=lon, alt=alt, hdg=hdg, tas=tas, vs=vs,
    )


def make_action() -> DiscreteAction:
    return DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)


@pytest.fixture()
def capacity_config() -> dict:
    """Minimal config with capacity penalty component."""
    return {
        "components": {
            "capacity": {
                "enabled": True,
                "weight": 1.0,
                "max_aircraft": 5,
                "penalty_per_excess": -10.0,
            },
        },
    }


class TestCapacityNoPenaltyUnderLimit:
    """No penalty when aircraft count is at or below the limit."""

    def test_capacity_no_penalty_under_limit(self, capacity_config: dict) -> None:
        """3 aircraft with limit 5 → no penalty."""
        comp = CapacityPenalty(capacity_config)
        own = make_state("OWN")
        all_states = {
            "OWN": own,
            "AC001": make_state("AC001", lat=39.30),
            "AC002": make_state("AC002", lat=39.35),
        }
        action = make_action()

        result = comp.compute("OWN", own, action, own, all_states)
        assert result == 0.0


class TestCapacityPenaltyOverLimit:
    """Penalty should be given when aircraft count exceeds the limit."""

    def test_capacity_penalty_over_limit(self, capacity_config: dict) -> None:
        """7 aircraft with limit 5 → penalty for 2 excess."""
        comp = CapacityPenalty(capacity_config)
        own = make_state("OWN")
        all_states = {"OWN": own}
        for i in range(6):
            all_states[f"AC{i:03d}"] = make_state(f"AC{i:03d}", lat=39.25 + i * 0.01)
        action = make_action()

        result = comp.compute("OWN", own, action, own, all_states)
        # 7 aircraft, limit 5, excess = 2, penalty = 2 * -10 = -20
        assert result == -20.0


class TestCapacityPenaltyProportional:
    """Penalty should be proportional to the number of excess aircraft."""

    def test_capacity_penalty_proportional(self, capacity_config: dict) -> None:
        """More excess aircraft → more negative penalty."""
        comp = CapacityPenalty(capacity_config)
        action = make_action()

        # 8 aircraft (excess 3)
        states_8 = {f"AC{i:03d}": make_state(f"AC{i:03d}") for i in range(8)}
        result_8 = comp.compute("AC000", states_8["AC000"], action, states_8["AC000"], states_8)

        # 10 aircraft (excess 5)
        states_10 = {f"AC{i:03d}": make_state(f"AC{i:03d}") for i in range(10)}
        result_10 = comp.compute("AC000", states_10["AC000"], action, states_10["AC000"], states_10)

        assert result_8 < 0
        assert result_10 < result_8  # More excess → more negative


class TestCapacityThresholdConfigurable:
    """Capacity threshold should be configurable."""

    def test_capacity_threshold_configurable(self, capacity_config: dict) -> None:
        """Changing max_aircraft changes when penalty kicks in."""
        # With max_aircraft=3, 4 aircraft should trigger penalty
        capacity_config["components"]["capacity"]["max_aircraft"] = 3
        comp = CapacityPenalty(capacity_config)
        own = make_state("OWN")
        all_states = {
            "OWN": own,
            "AC001": make_state("AC001"),
            "AC002": make_state("AC002"),
            "AC003": make_state("AC003"),
        }
        action = make_action()

        result = comp.compute("OWN", own, action, own, all_states)
        # 4 aircraft, limit 3, excess = 1
        assert result == -10.0


class TestCapacityPenaltyReset:
    """Reset should not crash (stateless component)."""

    def test_capacity_penalty_reset(self, capacity_config: dict) -> None:
        """reset() completes without error and compute still works."""
        comp = CapacityPenalty(capacity_config)
        comp.reset()

        own = make_state("OWN")
        all_states = {"OWN": own}
        action = make_action()
        result = comp.compute("OWN", own, action, own, all_states)
        assert result == 0.0
