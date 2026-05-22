"""Tests for efficiency reward component."""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


def make_state(
    acid: str = "AC001",
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


class TestStepPenalty:
    """Test per-step penalty."""

    def test_step_penalty(self, rewards_config: dict) -> None:
        """Every step incurs a small penalty of -0.01."""
        comp = EfficiencyReward(rewards_config)
        state = make_state()
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        assert result == pytest.approx(-0.01)


class TestNoDeviation:
    """Test zero deviation when on target."""

    def test_no_deviation(self, rewards_config: dict) -> None:
        """Aircraft at goal waypoint → no deviation penalty, but arrival reward."""
        comp = EfficiencyReward(rewards_config)
        state = make_state(lat=39.25, lon=116.25)
        comp.set_goal("AC001", lat=39.25, lon=116.25)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        # At goal: step_penalty(-0.01) + deviation(0) + arrival(+10) = 9.99
        assert result == pytest.approx(9.99)


class TestDeviationPenalty:
    """Test deviation penalty."""

    def test_deviation_penalty(self, rewards_config: dict) -> None:
        """Deviation of 10NM → proportional penalty."""
        comp = EfficiencyReward(rewards_config)
        state = make_state(lat=39.25, lon=116.25)
        # Goal exactly 10NM north
        comp.set_goal("AC001", lat=39.4165543515, lon=116.25)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        # deviation_penalty = -(10/50)*5 = -1.0
        # total = -0.01 + (-1.0) + 0 = -1.01
        assert result == pytest.approx(-1.01)

    def test_max_deviation_penalty(self, rewards_config: dict) -> None:
        """Deviation at MAX_DEVIATION(50NM) → max penalty -5."""
        comp = EfficiencyReward(rewards_config)
        state = make_state(lat=39.25, lon=116.25)
        # Goal exactly 50NM north
        comp.set_goal("AC001", lat=40.0827717574, lon=116.25)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        # deviation_penalty = -(50/50)*5 = -5.0
        # total = -0.01 + (-5.0) + 0 = -5.01
        assert result == pytest.approx(-5.01)

    def test_deviation_proportional(self, rewards_config: dict) -> None:
        """Penalty is proportional to deviation distance."""
        comp = EfficiencyReward(rewards_config)
        state = make_state(lat=39.25, lon=116.25)
        action = make_action()

        # 20NM away
        comp.set_goal("AC001", lat=39.5831087030, lon=116.25)
        r20 = comp.compute("AC001", state, action, state, {"AC001": state})

        # 40NM away
        comp.set_goal("AC001", lat=39.9162174059, lon=116.25)
        r40 = comp.compute("AC001", state, action, state, {"AC001": state})

        # Both negative, larger deviation → more negative
        assert r40 < r20 < 0


class TestArrivalReward:
    """Test arrival reward."""

    def test_arrival_reward(self, rewards_config: dict) -> None:
        """Arrive at goal waypoint → +10 reward."""
        comp = EfficiencyReward(rewards_config)
        state = make_state(lat=39.25, lon=116.25)
        comp.set_goal("AC001", lat=39.25, lon=116.25)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        # step_penalty(-0.01) + deviation(0) + arrival(+10) = 9.99
        assert result == pytest.approx(9.99)

    def test_not_arrived(self, rewards_config: dict) -> None:
        """Not arrived (far from goal) → no arrival reward."""
        comp = EfficiencyReward(rewards_config)
        state = make_state(lat=39.25, lon=116.25)
        # Goal 10NM away — well outside arrival threshold (2NM)
        comp.set_goal("AC001", lat=39.4165543515, lon=116.25)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        # No arrival reward, just step + deviation penalty
        assert result == pytest.approx(-1.01)
        assert result < 0  # Definitely no +10 arrival bonus
