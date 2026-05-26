"""Tests for simulation step frequency / step_n (T-V02)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper


# ---------------------------------------------------------------------------
# BlueSkyWrapper for testing step_n without real BlueSky
# ---------------------------------------------------------------------------


class TestStepNExecutesNTimes:
    """step_n(n) should execute exactly n simulation steps."""

    def test_step_n_executes_n_times(self) -> None:
        config = {"simulation": {"dt": 5.0}}
        wrapper = BlueSkyWrapper(config)
        wrapper.init_simulation()

        wrapper.step_n(5)
        assert wrapper._step_count == 5

        wrapper.step_n(3)
        assert wrapper._step_count == 8

    def test_step_n_zero(self) -> None:
        config = {"simulation": {"dt": 5.0}}
        wrapper = BlueSkyWrapper(config)
        wrapper.init_simulation()

        wrapper.step_n(0)
        assert wrapper._step_count == 0
        assert wrapper._simt == 0.0


class TestActionFrequencyConfigurable:
    """ACTION_FREQUENCY should be read from config."""

    def test_action_frequency_configurable(self) -> None:
        config = {"simulation": {"dt": 5.0, "action_frequency": 10}}
        wrapper = BlueSkyWrapper(config)
        wrapper.init_simulation()

        freq = config["simulation"].get("action_frequency", 1)
        wrapper.step_n(freq)

        assert wrapper._step_count == 10


class TestStateAfterMultipleSteps:
    """Aircraft state should correctly update after multiple steps."""

    def test_state_after_multiple_steps(self) -> None:
        config = {"simulation": {"dt": 5.0}}
        wrapper = BlueSkyWrapper(config)
        wrapper.init_simulation()

        wrapper.create_aircraft("AC000", "B737", 39.0, 116.0, 35000, 90.0, 450.0)

        # After 1 step with hdg=90 (east), longitude should increase
        wrapper.step_n(1)
        st1 = wrapper.get_aircraft_state("AC000")
        assert st1["lon"] > 116.0

        # After 10 more steps, should have moved further
        lon_after_1 = st1["lon"]
        wrapper.step_n(10)
        st11 = wrapper.get_aircraft_state("AC000")
        assert st11["lon"] > lon_after_1

    def test_heading_north_increases_lat(self) -> None:
        config = {"simulation": {"dt": 5.0}}
        wrapper = BlueSkyWrapper(config)
        wrapper.init_simulation()

        wrapper.create_aircraft("AC000", "B737", 39.0, 116.0, 35000, 0.0, 450.0)
        wrapper.step_n(1)
        st = wrapper.get_aircraft_state("AC000")
        assert st["lat"] > 39.0  # heading north → latitude increases

    def test_heading_south_decreases_lat(self) -> None:
        config = {"simulation": {"dt": 5.0}}
        wrapper = BlueSkyWrapper(config)
        wrapper.init_simulation()

        wrapper.create_aircraft("AC000", "B737", 39.0, 116.0, 35000, 180.0, 450.0)
        wrapper.step_n(1)
        st = wrapper.get_aircraft_state("AC000")
        assert st["lat"] < 39.0  # heading south → latitude decreases


class TestDefaultFrequencyIs1:
    """Default action_frequency should be 1 (backward compatible)."""

    def test_default_frequency_is_1(self) -> None:
        config = {"simulation": {"dt": 5.0}}  # No action_frequency key
        wrapper = BlueSkyWrapper(config)
        wrapper.init_simulation()

        # Without action_frequency, step() should be equivalent to step_n(1)
        wrapper.create_aircraft("AC000", "B737", 39.0, 116.0, 35000, 90.0, 450.0)

        wrapper.step()
        assert wrapper._step_count == 1
        assert wrapper._simt == pytest.approx(5.0)


class TestStepNMethodExists:
    """BlueSkyWrapper should have a step_n method."""

    def test_step_n_method_exists(self) -> None:
        assert hasattr(BlueSkyWrapper, "step_n"), "BlueSkyWrapper must have step_n method"

    def test_step_calls_step_n(self) -> None:
        """step() should delegate to step_n(1) for backward compatibility."""
        import inspect
        source = inspect.getsource(BlueSkyWrapper.step)
        assert "step_n" in source, "step() should call step_n() internally"


class TestTimeAdvancesCorrectly:
    """Simulation time should advance by dt * n after step_n(n)."""

    def test_time_advances_correctly(self) -> None:
        config = {"simulation": {"dt": 10.0}}
        wrapper = BlueSkyWrapper(config)
        wrapper.init_simulation()

        wrapper.step_n(3)
        assert wrapper._simt == pytest.approx(30.0)  # 10 * 3

        wrapper.step_n(5)
        assert wrapper._simt == pytest.approx(80.0)  # 30 + 10 * 5

    def test_time_with_different_dt(self) -> None:
        config = {"simulation": {"dt": 2.5}}
        wrapper = BlueSkyWrapper(config)
        wrapper.init_simulation()

        wrapper.step_n(4)
        assert wrapper._simt == pytest.approx(10.0)  # 2.5 * 4
