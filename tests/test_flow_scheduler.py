"""Tests for FlowScheduler — departure/arrival timing and sector handoff."""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.flow.scheduler import FlowScheduler


class TestFlowSchedulerCreation:
    """Test scheduler creation."""

    def test_create_default(self) -> None:
        sched = FlowScheduler({})
        assert sched is not None

    def test_create_with_config(self) -> None:
        config = {"flow": {"departure_interval": 3, "arrival_interval": 2}}
        sched = FlowScheduler(config)
        assert sched is not None


class TestDepartureSpacing:
    """Test departure interval enforcement."""

    def test_first_departure_always_allowed(self) -> None:
        sched = FlowScheduler({"flow": {"departure_interval": 3}})
        assert sched.check_departure("AC000", step=0) is True

    def test_departure_too_soon_rejected(self) -> None:
        sched = FlowScheduler({"flow": {"departure_interval": 3}})
        sched.check_departure("AC000", step=0)  # allow
        assert sched.check_departure("AC001", step=1) is False

    def test_departure_after_interval_allowed(self) -> None:
        sched = FlowScheduler({"flow": {"departure_interval": 3}})
        sched.check_departure("AC000", step=0)
        assert sched.check_departure("AC001", step=3) is True

    def test_departure_interval_1_allows_every_step(self) -> None:
        sched = FlowScheduler({"flow": {"departure_interval": 1}})
        assert sched.check_departure("AC000", step=0) is True
        assert sched.check_departure("AC001", step=1) is True
        assert sched.check_departure("AC002", step=2) is True

    def test_default_departure_interval(self) -> None:
        sched = FlowScheduler({})
        # Default interval should be 1 (no restriction)
        assert sched.check_departure("AC000", step=0) is True
        assert sched.check_departure("AC001", step=1) is True


class TestArrivalSpacing:
    """Test arrival interval enforcement."""

    def test_first_arrival_always_allowed(self) -> None:
        sched = FlowScheduler({"flow": {"arrival_interval": 2}})
        assert sched.check_arrival("AC000", step=5) is True

    def test_arrival_too_soon_rejected(self) -> None:
        sched = FlowScheduler({"flow": {"arrival_interval": 2}})
        sched.check_arrival("AC000", step=5)
        assert sched.check_arrival("AC001", step=6) is False

    def test_arrival_after_interval_allowed(self) -> None:
        sched = FlowScheduler({"flow": {"arrival_interval": 2}})
        sched.check_arrival("AC000", step=5)
        assert sched.check_arrival("AC001", step=7) is True

    def test_default_arrival_interval(self) -> None:
        sched = FlowScheduler({})
        assert sched.check_arrival("AC000", step=5) is True
        assert sched.check_arrival("AC001", step=6) is True


class TestSectorHandoff:
    """Test sector handoff tracking."""

    def test_notify_sector_change(self) -> None:
        sched = FlowScheduler({})
        sched.notify_sector_change("AC000", "S1", "S2")
        # No exception = pass

    def test_handoff_delays_recorded(self) -> None:
        sched = FlowScheduler({})
        sched.notify_sector_change("AC000", "S1", "S2")
        delays = sched.get_handoff_delays()
        assert "AC000" in delays

    def test_multiple_handoffs(self) -> None:
        sched = FlowScheduler({})
        sched.notify_sector_change("AC000", "S1", "S2")
        sched.notify_sector_change("AC000", "S2", "S3")
        sched.notify_sector_change("AC001", "S1", "S3")
        delays = sched.get_handoff_delays()
        assert len(delays) == 2

    def test_handoff_count_tracked(self) -> None:
        sched = FlowScheduler({})
        sched.notify_sector_change("AC000", "S1", "S2")
        sched.notify_sector_change("AC000", "S2", "S3")
        delays = sched.get_handoff_delays()
        # AC000 had 2 handoffs
        assert delays["AC000"] == 2


class TestReset:
    """Test reset clears all state."""

    def test_reset_clears_departures(self) -> None:
        sched = FlowScheduler({"flow": {"departure_interval": 100}})
        sched.check_departure("AC000", step=0)
        sched.reset()
        # After reset, first departure should be allowed again
        assert sched.check_departure("AC000", step=0) is True

    def test_reset_clears_arrivals(self) -> None:
        sched = FlowScheduler({"flow": {"arrival_interval": 100}})
        sched.check_arrival("AC000", step=5)
        sched.reset()
        assert sched.check_arrival("AC000", step=5) is True

    def test_reset_clears_handoffs(self) -> None:
        sched = FlowScheduler({})
        sched.notify_sector_change("AC000", "S1", "S2")
        sched.reset()
        assert sched.get_handoff_delays() == {}
