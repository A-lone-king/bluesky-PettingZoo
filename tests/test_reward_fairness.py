"""Tests for FairnessReward component."""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.rewards.components.fairness import FairnessReward
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


def _make_state(acid: str = "AC000") -> AircraftState:
    return AircraftState(id=acid, lat=40.0, lon=-74.0, alt=35000, hdg=90, tas=450, vs=0)


def _make_action() -> DiscreteAction:
    return DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)


class TestFairnessCreation:
    """Test component creation."""

    def test_create_default(self) -> None:
        comp = FairnessReward({})
        assert comp is not None

    def test_create_with_config(self) -> None:
        config = {"components": {"fairness": {"penalty_factor": 0.5}}}
        comp = FairnessReward(config)
        assert comp is not None


class TestSetDelays:
    """Test delay data injection."""

    def test_set_delays(self) -> None:
        comp = FairnessReward({})
        comp.set_delays({"AC000": 5.0, "AC001": 3.0})
        # No exception = pass

    def test_set_empty_delays(self) -> None:
        comp = FairnessReward({})
        comp.set_delays({})


class TestComputeFairness:
    """Test reward computation."""

    def test_no_delays_returns_zero(self) -> None:
        comp = FairnessReward({})
        state = _make_state()
        result = comp.compute("AC000", state, _make_action(), state, {"AC000": state})
        assert result == pytest.approx(0.0)

    def test_all_equal_delays_no_penalty(self) -> None:
        comp = FairnessReward({})
        comp.set_delays({"AC000": 5.0, "AC001": 5.0, "AC002": 5.0})
        state = _make_state()
        result = comp.compute("AC000", state, _make_action(), state, {"AC000": state})
        assert result == pytest.approx(0.0)

    def test_unequal_delays_penalty(self) -> None:
        comp = FairnessReward({"components": {"fairness": {"penalty_factor": 1.0}}})
        comp.set_delays({"AC000": 0.0, "AC001": 10.0})
        state = _make_state()
        result = comp.compute("AC000", state, _make_action(), state, {"AC000": state})
        assert result < 0

    def test_higher_inequality_higher_penalty(self) -> None:
        comp_low = FairnessReward({"components": {"fairness": {"penalty_factor": 1.0}}})
        comp_low.set_delays({"AC000": 5.0, "AC001": 6.0})
        state = _make_state()
        result_low = comp_low.compute("AC000", state, _make_action(), state, {"AC000": state})

        comp_high = FairnessReward({"components": {"fairness": {"penalty_factor": 1.0}}})
        comp_high.set_delays({"AC000": 0.0, "AC001": 20.0})
        result_high = comp_high.compute("AC000", state, _make_action(), state, {"AC000": state})

        assert result_high < result_low

    def test_configurable_penalty_factor(self) -> None:
        comp_1x = FairnessReward({"components": {"fairness": {"penalty_factor": 1.0}}})
        comp_1x.set_delays({"AC000": 0.0, "AC001": 10.0})
        state = _make_state()
        r1 = comp_1x.compute("AC000", state, _make_action(), state, {"AC000": state})

        comp_2x = FairnessReward({"components": {"fairness": {"penalty_factor": 2.0}}})
        comp_2x.set_delays({"AC000": 0.0, "AC001": 10.0})
        r2 = comp_2x.compute("AC000", state, _make_action(), state, {"AC000": state})

        # Higher factor → more negative penalty
        assert r2 < r1


class TestReset:
    """Test reset clears state."""

    def test_reset_clears_delays(self) -> None:
        comp = FairnessReward({})
        comp.set_delays({"AC000": 5.0, "AC001": 10.0})
        comp.reset()

        state = _make_state()
        result = comp.compute("AC000", state, _make_action(), state, {"AC000": state})
        assert result == pytest.approx(0.0)
