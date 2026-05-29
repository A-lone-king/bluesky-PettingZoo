"""Tests for delay penalty reward component."""

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


class TestDelayNoGoal:
    """Agent without a goal should return 0."""

    def test_no_goal_returns_zero(self, delay_config: dict) -> None:
        comp = DelayPenalty(delay_config)
        own = make_state()
        result = comp.compute("OWN", own, make_action(), own, {"OWN": own}, step_count=10)
        assert result == 0.0


class TestDelayBeforeDeadline:
    """Agent before expected arrival should return 0."""

    def test_before_deadline(self, delay_config: dict) -> None:
        comp = DelayPenalty(delay_config)
        comp.set_goal("OWN", distance_nm=100.0, speed_kt=450.0, dt=5.0)
        own = make_state()
        # expected_steps = 100 / (450 * 5/3600) = 100 / 0.625 = 160 steps
        result = comp.compute("OWN", own, make_action(), own, {"OWN": own}, step_count=50)
        assert result == 0.0


class TestDelayAfterDeadline:
    """Agent past expected arrival should be penalized."""

    def test_after_deadline(self, delay_config: dict) -> None:
        comp = DelayPenalty(delay_config)
        comp.set_goal("OWN", distance_nm=100.0, speed_kt=450.0, dt=5.0)
        own = make_state()
        # expected_steps = 160; at step 170 → overdue by 10
        result = comp.compute("OWN", own, make_action(), own, {"OWN": own}, step_count=170)
        assert result < 0
        # -0.05 * 10 = -0.5
        assert abs(result - (-0.5)) < 0.01

    def test_penalty_proportional_to_overdue(self, delay_config: dict) -> None:
        comp = DelayPenalty(delay_config)
        comp.set_goal("OWN", distance_nm=100.0, speed_kt=450.0, dt=5.0)
        own = make_state()
        r1 = comp.compute("OWN", own, make_action(), own, {"OWN": own}, step_count=170)
        r2 = comp.compute("OWN", own, make_action(), own, {"OWN": own}, step_count=180)
        assert r2 < r1  # more overdue → more penalty


class TestDelayConfigurable:
    """Penalty per step should be configurable."""

    def test_custom_penalty(self, delay_config: dict) -> None:
        delay_config["components"]["delay"]["delay_penalty_per_step"] = -0.1
        comp = DelayPenalty(delay_config)
        comp.set_goal("OWN", distance_nm=10.0, speed_kt=450.0, dt=5.0)
        own = make_state()
        # expected = 10 / 0.625 = 16 steps; at step 20 → overdue 4
        result = comp.compute("OWN", own, make_action(), own, {"OWN": own}, step_count=20)
        assert abs(result - (-0.4)) < 0.01


class TestDelayMultipleAgents:
    """Different agents can have different goals."""

    def test_independent_agents(self, delay_config: dict) -> None:
        comp = DelayPenalty(delay_config)
        comp.set_goal("A", distance_nm=100.0, speed_kt=450.0, dt=5.0)  # 160 steps
        comp.set_goal("B", distance_nm=10.0, speed_kt=450.0, dt=5.0)   # 16 steps
        a = make_state("A")
        b = make_state("B")
        all_states = {"A": a, "B": b}
        action = make_action()
        # At step 20: A on time, B overdue by 4
        assert comp.compute("A", a, action, a, all_states, step_count=20) == 0.0
        assert comp.compute("B", b, action, b, all_states, step_count=20) < 0


class TestDelayReset:
    """Reset should clear all goals and step counts."""

    def test_reset_clears_state(self, delay_config: dict) -> None:
        comp = DelayPenalty(delay_config)
        comp.set_goal("OWN", distance_nm=10.0, speed_kt=450.0, dt=5.0)
        comp.reset()
        own = make_state()
        result = comp.compute("OWN", own, make_action(), own, {"OWN": own}, step_count=100)
        assert result == 0.0  # goal was cleared
