"""Tests for reward calculator (component registry and weighted sum)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bluesky_pettingzoo.rewards.base import RewardComponent
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


def make_state() -> AircraftState:
    return AircraftState(
        id="AC001", lat=39.25, lon=116.25, alt=35000.0,
        hdg=90.0, tas=450.0, vs=0.0,
    )


def make_action() -> DiscreteAction:
    return DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)


def make_mock_component(value: float) -> RewardComponent:
    """Create a mock reward component that returns a fixed value."""
    comp = MagicMock(spec=RewardComponent)
    comp.compute.return_value = value
    return comp


class TestRegister:
    """Test component registration."""

    def test_register_component(self) -> None:
        """Registering a component increases the components list."""
        calc = RewardCalculator()
        comp = make_mock_component(1.0)

        calc.register(comp, weight=1.0)

        assert len(calc.components) == 1


class TestEmptyCalculator:
    """Test calculator with no components."""

    def test_empty_calculator(self) -> None:
        """No components → returns 0."""
        calc = RewardCalculator()
        state = make_state()
        action = make_action()

        result = calc.compute("AC001", state, action, state, {"AC001": state})

        assert result == 0.0


class TestSingleComponent:
    """Test single component reward."""

    def test_single_component(self) -> None:
        """Single component returns value × weight."""
        calc = RewardCalculator()
        comp = make_mock_component(3.0)
        calc.register(comp, weight=2.0)
        state = make_state()
        action = make_action()

        result = calc.compute("AC001", state, action, state, {"AC001": state})

        assert result == pytest.approx(6.0)


class TestMultipleComponents:
    """Test multiple component weighted sum."""

    def test_multiple_components(self) -> None:
        """Multiple components return weighted sum."""
        calc = RewardCalculator()
        comp1 = make_mock_component(2.0)
        comp2 = make_mock_component(-5.0)
        calc.register(comp1, weight=1.0)
        calc.register(comp2, weight=0.5)
        state = make_state()
        action = make_action()

        result = calc.compute("AC001", state, action, state, {"AC001": state})

        # 1.0*2.0 + 0.5*(-5.0) = 2.0 - 2.5 = -0.5
        assert result == pytest.approx(-0.5)


class TestZeroWeight:
    """Test zero weight component."""

    def test_zero_weight(self) -> None:
        """Zero weight → no contribution from that component."""
        calc = RewardCalculator()
        comp1 = make_mock_component(10.0)
        comp2 = make_mock_component(999.0)
        calc.register(comp1, weight=1.0)
        calc.register(comp2, weight=0.0)
        state = make_state()
        action = make_action()

        result = calc.compute("AC001", state, action, state, {"AC001": state})

        assert result == pytest.approx(10.0)


class TestNegativeWeight:
    """Test negative weight component."""

    def test_negative_weight(self) -> None:
        """Negative weight flips the sign."""
        calc = RewardCalculator()
        comp = make_mock_component(5.0)
        calc.register(comp, weight=-2.0)
        state = make_state()
        action = make_action()

        result = calc.compute("AC001", state, action, state, {"AC001": state})

        assert result == pytest.approx(-10.0)


class TestReset:
    """Test reset propagation."""

    def test_reset_calls_components(self) -> None:
        """reset() calls reset() on all registered components."""
        calc = RewardCalculator()
        comp1 = make_mock_component(1.0)
        comp2 = make_mock_component(2.0)
        calc.register(comp1, weight=1.0)
        calc.register(comp2, weight=1.0)

        calc.reset()

        comp1.reset.assert_called_once()
        comp2.reset.assert_called_once()


class TestComputeArgs:
    """Test argument forwarding."""

    def test_compute_passes_correct_args(self) -> None:
        """compute() forwards all arguments to each component."""
        calc = RewardCalculator()
        comp = make_mock_component(0.0)
        calc.register(comp, weight=1.0)

        state = make_state()
        action = make_action()
        all_states = {"AC001": state}

        calc.compute("AC001", state, action, state, all_states)

        comp.compute.assert_called_once_with("AC001", state, action, state, all_states)
