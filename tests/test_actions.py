"""Tests for action translator module."""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.actions.translator import ActionTranslator
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


class TestNoAdjustment:
    """Test zero-adjustment action."""

    def test_no_adjustment(self, default_config: dict) -> None:
        """Index [2,2,2] (all zero) should produce no commands."""
        translator = ActionTranslator(default_config)
        state = make_state("AC001")
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)

        commands = translator.translate("AC001", state, action)

        assert commands == []


class TestHeadingAdjustment:
    """Test heading adjustments."""

    def test_heading_positive(self, default_config: dict) -> None:
        """Heading +10° → HDG AC001 {current+10}."""
        translator = ActionTranslator(default_config)
        state = make_state("AC001", hdg=90.0)
        action = DiscreteAction(heading_idx=3, altitude_idx=2, speed_idx=2)

        commands = translator.translate("AC001", state, action)

        assert len(commands) == 1
        assert commands[0] == "HDG AC001 100"

    def test_heading_negative(self, default_config: dict) -> None:
        """Heading -20° → HDG AC001 {current-20}."""
        translator = ActionTranslator(default_config)
        state = make_state("AC001", hdg=90.0)
        action = DiscreteAction(heading_idx=0, altitude_idx=2, speed_idx=2)

        commands = translator.translate("AC001", state, action)

        assert len(commands) == 1
        assert commands[0] == "HDG AC001 70"

    def test_heading_wraparound(self, default_config: dict) -> None:
        """Heading 350° +20° → 10° (wraparound)."""
        translator = ActionTranslator(default_config)
        state = make_state("AC001", hdg=350.0)
        action = DiscreteAction(heading_idx=4, altitude_idx=2, speed_idx=2)

        commands = translator.translate("AC001", state, action)

        assert len(commands) == 1
        assert commands[0] == "HDG AC001 10"


class TestAltitudeAdjustment:
    """Test altitude adjustments."""

    def test_altitude_up(self, default_config: dict) -> None:
        """Altitude +1000ft → ALT AC001 {current+1000}."""
        translator = ActionTranslator(default_config)
        state = make_state("AC001", alt=35000.0)
        action = DiscreteAction(heading_idx=2, altitude_idx=3, speed_idx=2)

        commands = translator.translate("AC001", state, action)

        assert len(commands) == 1
        assert commands[0] == "ALT AC001 36000"

    def test_altitude_down(self, default_config: dict) -> None:
        """Altitude -2000ft → ALT AC001 {current-2000}."""
        translator = ActionTranslator(default_config)
        state = make_state("AC001", alt=35000.0)
        action = DiscreteAction(heading_idx=2, altitude_idx=0, speed_idx=2)

        commands = translator.translate("AC001", state, action)

        assert len(commands) == 1
        assert commands[0] == "ALT AC001 33000"


class TestSpeedAdjustment:
    """Test speed adjustments."""

    def test_speed_up(self, default_config: dict) -> None:
        """Speed +10kt → SPD AC001 {current+10}."""
        translator = ActionTranslator(default_config)
        state = make_state("AC001", tas=450.0)
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=3)

        commands = translator.translate("AC001", state, action)

        assert len(commands) == 1
        assert commands[0] == "SPD AC001 460"

    def test_speed_down(self, default_config: dict) -> None:
        """Speed -20kt → SPD AC001 {current-20}."""
        translator = ActionTranslator(default_config)
        state = make_state("AC001", tas=450.0)
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=0)

        commands = translator.translate("AC001", state, action)

        assert len(commands) == 1
        assert commands[0] == "SPD AC001 430"


class TestCombinedAction:
    """Test combined multi-axis adjustments."""

    def test_combined_action(self, default_config: dict) -> None:
        """Adjusting heading+altitude+speed should produce 3 commands."""
        translator = ActionTranslator(default_config)
        state = make_state("AC001", hdg=90.0, alt=35000.0, tas=450.0)
        action = DiscreteAction(heading_idx=3, altitude_idx=3, speed_idx=3)

        commands = translator.translate("AC001", state, action)

        assert len(commands) == 3
        cmd_types = {c.split()[0] for c in commands}
        assert cmd_types == {"HDG", "ALT", "SPD"}


class TestTranslateBatch:
    """Test batch translation."""

    def test_translate_batch(self, default_config: dict) -> None:
        """Batch translation should merge all commands."""
        translator = ActionTranslator(default_config)

        actions = {
            "AC001": DiscreteAction(heading_idx=3, altitude_idx=2, speed_idx=2),
            "AC002": DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=0),
        }
        states = {
            "AC001": make_state("AC001", hdg=90.0),
            "AC002": make_state("AC002", tas=450.0),
        }

        commands = translator.translate_batch(actions, states)

        assert len(commands) == 2
        assert any("AC001" in c for c in commands)
        assert any("AC002" in c for c in commands)


class TestCommandFormat:
    """Test BlueSky command format."""

    def test_command_format(self, default_config: dict) -> None:
        """Commands should follow TYPE ACID VALUE format."""
        translator = ActionTranslator(default_config)
        state = make_state("AC001", hdg=90.0)
        action = DiscreteAction(heading_idx=3, altitude_idx=2, speed_idx=2)

        commands = translator.translate("AC001", state, action)

        for cmd in commands:
            parts = cmd.split()
            assert len(parts) == 3
            assert parts[0] in ("HDG", "ALT", "SPD")
            assert parts[1] == "AC001"
            float(parts[2])  # should not raise
