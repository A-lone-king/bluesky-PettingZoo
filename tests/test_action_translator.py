"""Tests for unified ActionTranslator (spec4 F2).

Verify that ActionTranslator can dispatch to discrete or continuous translator.
"""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


def make_state(hdg: float = 90.0, alt: float = 35000.0, tas: float = 450.0) -> AircraftState:
    return AircraftState(
        id="AC001",
        lat=39.25,
        lon=116.25,
        alt=alt,
        hdg=hdg,
        tas=tas,
        vs=0.0,
    )


@pytest.fixture
def config() -> dict:
    return {
        "action": {
            "heading_adjustments": [-20, -10, 0, 10, 20],
            "altitude_adjustments": [-2000, -1000, 0, 1000, 2000],
            "speed_adjustments": [-20, -10, 0, 10, 20],
        },
        "continuous_action": {
            "heading_scale": 45.0,
            "altitude_scale": 12.5,
            "speed_scale": 6.67,
        },
    }


class TestDiscreteTranslation:
    """ActionTranslator should handle discrete actions."""

    def test_discrete_translate(self, config: dict) -> None:
        """translate() should work with DiscreteAction."""
        translator = ActionTranslator(config)
        state = make_state(hdg=90.0)
        action = DiscreteAction(heading_idx=3, altitude_idx=2, speed_idx=2)

        commands = translator.translate("AC001", state, action)

        assert len(commands) == 1
        assert "HDG" in commands[0]

    def test_discrete_noop(self, config: dict) -> None:
        """Noop action (all 2s) should produce no commands."""
        translator = ActionTranslator(config)
        state = make_state()
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)

        commands = translator.translate("AC001", state, action)

        assert len(commands) == 0


class TestContinuousTranslation:
    """ActionTranslator should handle continuous actions."""

    def test_continuous_translate(self, config: dict) -> None:
        """translate_continuous() should work with array action."""
        translator = ActionTranslator(config)
        state = make_state(hdg=90.0)
        action = np.array([0.5, 0.0, 0.0], dtype=np.float32)

        commands = translator.translate_continuous("AC001", state, action)

        assert len(commands) == 1
        assert "HDG" in commands[0]

    def test_continuous_noop(self, config: dict) -> None:
        """Zero continuous action should produce no commands."""
        translator = ActionTranslator(config)
        state = make_state()
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        commands = translator.translate_continuous("AC001", state, action)

        assert len(commands) == 0

    def test_continuous_batch(self, config: dict) -> None:
        """translate_continuous_batch() should handle multiple agents."""
        translator = ActionTranslator(config)
        states = {
            "AC001": make_state(hdg=90.0),
            "AC002": make_state(hdg=180.0),
        }
        actions = {
            "AC001": np.array([0.5, 0.0, 0.0], dtype=np.float32),
            "AC002": np.array([-0.5, 0.0, 0.0], dtype=np.float32),
        }

        commands = translator.translate_continuous_batch(actions, states)

        assert len(commands) == 2
        assert any("AC001" in c for c in commands)
        assert any("AC002" in c for c in commands)
