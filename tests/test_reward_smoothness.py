"""Tests for smoothness penalty reward component."""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from bluesky_pettingzoo.utils.types import DiscreteAction
from tests.helpers.state_factory import make_action, make_state


class TestNoActionPenalty:
    """Test zero-adjustment action."""

    def test_no_action_penalty(self, rewards_config: dict) -> None:
        """Zero adjustment [2,2,2] should return 0."""
        comp = SmoothnessPenalty(rewards_config)
        state = make_state()
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        assert result == 0.0


class TestHeadingPenalty:
    """Test heading adjustment penalty."""

    def test_heading_action_penalty(self, rewards_config: dict) -> None:
        """Heading adjustment should return action_penalty."""
        comp = SmoothnessPenalty(rewards_config)
        state = make_state()
        action = DiscreteAction(heading_idx=3, altitude_idx=2, speed_idx=2)

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        assert result == pytest.approx(-0.1)


class TestAltitudePenalty:
    """Test altitude adjustment penalty."""

    def test_altitude_action_penalty(self, rewards_config: dict) -> None:
        """Altitude adjustment should return action_penalty."""
        comp = SmoothnessPenalty(rewards_config)
        state = make_state()
        action = DiscreteAction(heading_idx=2, altitude_idx=0, speed_idx=2)

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        assert result == pytest.approx(-0.1)


class TestSpeedPenalty:
    """Test speed adjustment penalty."""

    def test_speed_action_penalty(self, rewards_config: dict) -> None:
        """Speed adjustment should return action_penalty."""
        comp = SmoothnessPenalty(rewards_config)
        state = make_state()
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=4)

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        assert result == pytest.approx(-0.1)


class TestCombinedPenalty:
    """Test combined multi-axis adjustment."""

    def test_combined_action_penalty(self, rewards_config: dict) -> None:
        """Multi-axis adjustment should still return single action_penalty."""
        comp = SmoothnessPenalty(rewards_config)
        state = make_state()
        action = DiscreteAction(heading_idx=0, altitude_idx=4, speed_idx=3)

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        assert result == pytest.approx(-0.1)
