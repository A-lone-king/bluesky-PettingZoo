"""Tests for efficiency reward aligned with bluesky-gym (spec4 F1).

Verify that arrival_reward=1.0, step_penalty=0.0, deviation_scale=0.0.
"""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


def make_state(lat: float = 39.25, lon: float = 116.25) -> AircraftState:
    return AircraftState(
        id="AC001",
        lat=lat,
        lon=lon,
        alt=35000.0,
        hdg=90.0,
        tas=450.0,
        vs=0.0,
    )


def make_action() -> DiscreteAction:
    return DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)


@pytest.fixture
def config() -> dict:
    return {
        "components": {
            "efficiency": {
                "enabled": True,
                "weight": 0.3,
                "max_deviation_nm": 50,
                "deviation_penalty_scale": 0.0,
                "arrival_reward": 1.0,
                "step_penalty": 0.0,
                "arrival_threshold_nm": 2,
            }
        }
    }


class TestArrivalReward:
    """Arrival reward should be 1.0 (aligns with bluesky-gym +1)."""

    def test_arrival_reward_value(self, config: dict) -> None:
        comp = EfficiencyReward(config)
        comp.set_goal("AC001", 39.25, 116.25)
        state = make_state(lat=39.25, lon=116.25)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        assert result == pytest.approx(1.0)

    def test_arrival_reward_at_threshold(self, config: dict) -> None:
        """Should get arrival reward when within threshold."""
        comp = EfficiencyReward(config)
        comp.set_goal("AC001", 39.25, 116.25)
        state = make_state(lat=39.25, lon=116.26)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        assert result == pytest.approx(1.0)


class TestStepPenalty:
    """Step penalty should be 0.0 (aligns with bluesky-gym no step penalty)."""

    def test_step_penalty_zero(self, config: dict) -> None:
        comp = EfficiencyReward(config)
        comp.set_goal("AC001", 39.30, 116.30)
        state = make_state(lat=39.25, lon=116.25)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        # Should not include step penalty
        # Only deviation penalty (which is 0 due to deviation_scale=0)
        assert result == pytest.approx(0.0)

    def test_no_step_penalty_without_goal(self, config: dict) -> None:
        """Without goal, should return 0.0 (no step penalty)."""
        comp = EfficiencyReward(config)
        state = make_state()
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        assert result == pytest.approx(0.0)


class TestDeviationScale:
    """Deviation penalty scale should be 0.0 (disabled)."""

    def test_deviation_penalty_disabled(self, config: dict) -> None:
        comp = EfficiencyReward(config)
        comp.set_goal("AC001", 39.30, 116.30)
        state = make_state(lat=39.25, lon=116.25)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        # No deviation penalty, no step penalty, no arrival
        assert result == pytest.approx(0.0)


class TestNoArrivalWhenFar:
    """Should not get arrival reward when far from goal."""

    def test_no_arrival_when_far(self, config: dict) -> None:
        comp = EfficiencyReward(config)
        comp.set_goal("AC001", 39.30, 116.30)
        state = make_state(lat=39.25, lon=116.25)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        assert result != pytest.approx(1.0)
