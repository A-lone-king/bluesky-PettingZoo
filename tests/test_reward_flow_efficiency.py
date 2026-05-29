"""Tests for FlowEfficiencyReward component."""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.rewards.components.flow_efficiency import FlowEfficiencyReward
from tests.helpers.state_factory import make_action, make_state


class TestFlowEfficiencyCreation:
    """Test component creation and defaults."""

    def test_create_default(self) -> None:
        comp = FlowEfficiencyReward({})
        assert comp is not None

    def test_create_with_config(self) -> None:
        config = {"components": {"flow_efficiency": {"reward_per_aircraft": 0.5}}}
        comp = FlowEfficiencyReward(config)
        assert comp is not None


class TestNotifySectorEntry:
    """Test sector entry notification."""

    def test_notify_records_entry(self) -> None:
        comp = FlowEfficiencyReward({})
        comp.notify_sector_entry("AC000", "S1")
        # No exception = pass; entry is recorded internally

    def test_multiple_entries_same_aircraft(self) -> None:
        comp = FlowEfficiencyReward({})
        comp.notify_sector_entry("AC000", "S1")
        comp.notify_sector_entry("AC000", "S2")
        # Each entry is a separate event

    def test_multiple_aircraft_same_sector(self) -> None:
        comp = FlowEfficiencyReward({})
        comp.notify_sector_entry("AC000", "S1")
        comp.notify_sector_entry("AC001", "S1")
        comp.notify_sector_entry("AC002", "S1")


class TestComputeFlowEfficiency:
    """Test reward computation."""

    def test_no_entries_returns_zero(self) -> None:
        comp = FlowEfficiencyReward({})
        state = make_state()
        result = comp.compute("AC000", state, make_action(), state, {"AC000": state})
        assert result == pytest.approx(0.0)

    def test_single_entry_returns_positive(self) -> None:
        comp = FlowEfficiencyReward({"components": {"flow_efficiency": {"reward_per_aircraft": 1.0}}})
        comp.notify_sector_entry("AC000", "S1")
        state = make_state()
        result = comp.compute("AC000", state, make_action(), state, {"AC000": state})
        assert result > 0

    def test_more_entries_higher_reward(self) -> None:
        comp = FlowEfficiencyReward({"components": {"flow_efficiency": {"reward_per_aircraft": 1.0}}})
        comp.notify_sector_entry("AC000", "S1")
        comp.notify_sector_entry("AC001", "S1")
        comp.notify_sector_entry("AC002", "S1")
        state = make_state()
        all_states = {"AC000": state, "AC001": make_state("AC001"), "AC002": make_state("AC002")}
        result_3 = comp.compute("AC000", state, make_action(), state, all_states)

        comp2 = FlowEfficiencyReward({"components": {"flow_efficiency": {"reward_per_aircraft": 1.0}}})
        comp2.notify_sector_entry("AC000", "S1")
        result_1 = comp2.compute("AC000", state, make_action(), state, {"AC000": state})

        assert result_3 > result_1

    def test_configurable_reward_per_aircraft(self) -> None:
        comp = FlowEfficiencyReward({"components": {"flow_efficiency": {"reward_per_aircraft": 2.0}}})
        comp.notify_sector_entry("AC000", "S1")
        state = make_state()
        result = comp.compute("AC000", state, make_action(), state, {"AC000": state})
        assert result == pytest.approx(2.0)


class TestReset:
    """Test reset clears state."""

    def test_reset_clears_entries(self) -> None:
        comp = FlowEfficiencyReward({})
        comp.notify_sector_entry("AC000", "S1")
        comp.notify_sector_entry("AC001", "S1")
        comp.reset()

        state = make_state()
        result = comp.compute("AC000", state, make_action(), state, {"AC000": state})
        assert result == pytest.approx(0.0)
