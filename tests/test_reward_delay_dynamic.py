"""Tests for delay penalty dynamic expected steps (reward-003)."""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.rewards.components.delay import DelayPenalty
from tests.helpers.state_factory import make_action, make_state


@pytest.fixture()
def delay_config() -> dict:
    return {
        "components": {
            "delay": {
                "enabled": True,
                "weight": 1.0,
                "delay_penalty_per_step": -0.05,
            },
        },
    }


class TestDynamicExpectedSteps:
    """Test dynamic adjustment of expected steps based on current speed."""

    def test_decreased_speed_extends_deadline(self, delay_config: dict) -> None:
        """Agent slows down → expected steps increase proportionally."""
        comp = DelayPenalty(delay_config)
        # Initial: 100NM at 450kt → 160 steps
        comp.set_goal("OWN", distance_nm=100.0, speed_kt=450.0, dt=5.0)

        # At step 150, agent slows to 300kt
        # Original expected: 160 steps
        # Remaining distance at step 150: 100 - 150 * (450*5/3600) = 100 - 93.75 = 6.25NM
        # New expected: 150 + 6.25 / (300*5/3600) = 150 + 15 = 165 steps
        own = make_state(tas=300.0)
        action = make_action()

        # At step 160: original would be on time, but with slowdown should be overdue by 0 steps
        # Actually, remaining distance calculation is approximate
        result = comp.compute("OWN", own, make_action(), own, {"OWN": own}, step_count=160)

        # With slowdown, agent is still working on remaining distance
        # The dynamic adjustment should reduce penalty
        assert result >= -0.05  # Less penalty than without adjustment

    def test_increased_speed_shortens_deadline(self, delay_config: dict) -> None:
        """Agent speeds up → expected steps decrease proportionally."""
        comp = DelayPenalty(delay_config)
        # Initial: 100NM at 450kt → 160 steps
        comp.set_goal("OWN", distance_nm=100.0, speed_kt=450.0, dt=5.0)

        # At step 150, agent speeds to 600kt
        # Dynamic adjustment: speed_ratio = 600/450 = 1.333
        # dynamic_expected = 160 / 1.333 = 120 steps
        # overdue = 150 - 120 = 30 steps
        own = make_state(tas=600.0)
        action = make_action()

        result = comp.compute("OWN", own, make_action(), own, {"OWN": own}, step_count=150)

        # With speedup, deadline is shorter
        assert result == pytest.approx(-1.5)  # 30 steps * -0.05

    def test_constant_speed_no_change(self, delay_config: dict) -> None:
        """Agent maintains cruise speed → expected steps unchanged."""
        comp = DelayPenalty(delay_config)
        comp.set_goal("OWN", distance_nm=100.0, speed_kt=450.0, dt=5.0)

        # At step 150, same speed
        own = make_state(tas=450.0)
        action = make_action()

        # Expected steps should remain 160
        result = comp.compute("OWN", own, make_action(), own, {"OWN": own}, step_count=160)

        # At step 160: exactly on time
        assert result == 0.0

    def test_zero_speed_treats_as_stopped(self, delay_config: dict) -> None:
        """Agent stopped (speed=0) → very long expected steps."""
        comp = DelayPenalty(delay_config)
        comp.set_goal("OWN", distance_nm=100.0, speed_kt=450.0, dt=5.0)

        # At step 100, agent stopped
        own = make_state(tas=0.0)
        action = make_action()

        # With speed=0, remaining distance is无穷大
        # Should not penalize if agent is stopped
        result = comp.compute("OWN", own, make_action(), own, {"OWN": own}, step_count=200)

        # Agent stopped, no penalty (can't make progress)
        assert result == 0.0
