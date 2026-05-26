"""Tests for drift penalty reward component (spec4 F1).

DriftPenalty computes: scale * |heading - bearing_to_goal| in radians.
This aligns with bluesky-gym's -0.1 * |drift_radians| pattern.
"""

from __future__ import annotations

import math

import pytest

from bluesky_pettingzoo.rewards.components.drift import DriftPenalty
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


def make_state(lat: float = 39.25, lon: float = 116.25, hdg: float = 90.0) -> AircraftState:
    return AircraftState(
        id="AC001", lat=lat, lon=lon, alt=35000.0,
        hdg=hdg, tas=450.0, vs=0.0,
    )


def make_action() -> DiscreteAction:
    return DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)


@pytest.fixture
def config() -> dict:
    return {
        "components": {
            "drift_penalty": {
                "enabled": True,
                "weight": 0.5,
                "scale": -0.1,
            }
        }
    }


class TestDriftPenaltyNoGoal:
    """DriftPenalty should return 0.0 when no goal is set."""

    def test_no_goal_returns_zero(self, config: dict) -> None:
        comp = DriftPenalty(config)
        state = make_state()
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        assert result == 0.0


class TestDriftPenaltyWithGoal:
    """DriftPenalty should compute drift angle penalty when goal is set."""

    def test_heading_towards_goal_returns_zero(self, config: dict) -> None:
        """When heading points directly at goal, drift = 0, penalty = 0."""
        comp = DriftPenalty(config)
        # Goal is east of current position, heading is east (90°)
        comp.set_goal("AC001", {"lat": 39.25, "lon": 116.35})
        state = make_state(hdg=90.0)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        assert result == pytest.approx(0.0, abs=1e-4)

    def test_heading_away_from_goal_returns_penalty(self, config: dict) -> None:
        """When heading points away from goal, drift ≈ π, penalty ≈ -0.1 * π."""
        comp = DriftPenalty(config)
        # Goal is east of current position, heading is west (270°)
        comp.set_goal("AC001", {"lat": 39.25, "lon": 116.35})
        state = make_state(hdg=270.0)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        # drift ≈ π radians, penalty ≈ -0.1 * π ≈ -0.314
        assert result == pytest.approx(-0.1 * math.pi, abs=0.01)

    def test_perpendicular_heading_returns_penalty(self, config: dict) -> None:
        """When heading is perpendicular to goal, drift ≈ π/2."""
        comp = DriftPenalty(config)
        # Goal is east of current position, heading is north (0°)
        comp.set_goal("AC001", {"lat": 39.25, "lon": 116.35})
        state = make_state(hdg=0.0)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        # drift ≈ π/2 radians, penalty ≈ -0.1 * π/2 ≈ -0.157
        assert result == pytest.approx(-0.1 * math.pi / 2, abs=0.01)


class TestDriftPenaltyGoalCache:
    """DriftPenalty should cache goals per agent."""

    def test_set_goal_caches_value(self, config: dict) -> None:
        comp = DriftPenalty(config)
        goal = {"lat": 39.30, "lon": 116.30}
        comp.set_goal("AC001", goal)

        assert comp._goal_cache["AC001"] == goal

    def test_different_agents_different_goals(self, config: dict) -> None:
        comp = DriftPenalty(config)
        comp.set_goal("AC001", {"lat": 39.30, "lon": 116.30})
        comp.set_goal("AC002", {"lat": 39.20, "lon": 116.20})

        assert comp._goal_cache["AC001"] != comp._goal_cache["AC002"]

    def test_reset_clears_cache(self, config: dict) -> None:
        comp = DriftPenalty(config)
        comp.set_goal("AC001", {"lat": 39.30, "lon": 116.30})
        comp.reset()

        assert len(comp._goal_cache) == 0


class TestDriftPenaltyScale:
    """DriftPenalty scale should be configurable."""

    def test_custom_scale(self, config: dict) -> None:
        config["components"]["drift_penalty"]["scale"] = -0.5
        comp = DriftPenalty(config)
        comp.set_goal("AC001", {"lat": 39.25, "lon": 116.35})
        state = make_state(hdg=270.0)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        # drift ≈ π, penalty ≈ -0.5 * π
        assert result == pytest.approx(-0.5 * math.pi, abs=0.01)
