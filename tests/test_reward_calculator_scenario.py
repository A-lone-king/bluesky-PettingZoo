"""Tests for RewardCalculator scenario overrides (spec4 F1).

Verify that scenario reward_overrides can override default config values.
"""

from __future__ import annotations

import math

import pytest

from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.drift import DriftPenalty
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


def make_state(hdg: float = 90.0) -> AircraftState:
    return AircraftState(
        id="AC001",
        lat=39.25,
        lon=116.25,
        alt=35000.0,
        hdg=hdg,
        tas=450.0,
        vs=0.0,
    )


def make_action() -> DiscreteAction:
    return DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)


@pytest.fixture
def base_config() -> dict:
    return {
        "components": {
            "drift_penalty": {
                "enabled": True,
                "weight": 0.5,
                "scale": -0.1,
            }
        }
    }


class TestMergeRewardConfig:
    """RewardCalculator.merge_scenario_overrides() should merge configs."""

    def test_merge_overrides_scale(self, base_config: dict) -> None:
        """Scenario overrides should override base config scale."""
        scenario_overrides = {
            "drift_penalty": {
                "scale": -0.5,
            }
        }

        merged = RewardCalculator.merge_reward_config(base_config, scenario_overrides)

        assert merged["components"]["drift_penalty"]["scale"] == pytest.approx(-0.5)
        # Weight should remain from base config
        assert merged["components"]["drift_penalty"]["weight"] == pytest.approx(0.5)

    def test_merge_overrides_weight(self, base_config: dict) -> None:
        """Scenario can override weight."""
        scenario_overrides = {
            "drift_penalty": {
                "weight": 0.8,
            }
        }

        merged = RewardCalculator.merge_reward_config(base_config, scenario_overrides)

        assert merged["components"]["drift_penalty"]["weight"] == pytest.approx(0.8)

    def test_no_overrides_preserves_base(self, base_config: dict) -> None:
        """Empty overrides should preserve base config."""
        merged = RewardCalculator.merge_reward_config(base_config, {})

        assert merged["components"]["drift_penalty"]["scale"] == pytest.approx(-0.1)
        assert merged["components"]["drift_penalty"]["weight"] == pytest.approx(0.5)

    def test_merge_does_not_mutate_base(self, base_config: dict) -> None:
        """Merge should not mutate the original base config."""
        scenario_overrides = {
            "drift_penalty": {
                "scale": -0.5,
            }
        }

        original_scale = base_config["components"]["drift_penalty"]["scale"]
        RewardCalculator.merge_reward_config(base_config, scenario_overrides)

        assert base_config["components"]["drift_penalty"]["scale"] == pytest.approx(original_scale)


class TestRewardCalculatorWithOverrides:
    """RewardCalculator should work with merged config."""

    def test_calculator_uses_merged_config(self, base_config: dict) -> None:
        """Calculator should use merged config for component initialization."""
        scenario_overrides = {
            "drift_penalty": {
                "scale": -0.5,
            }
        }
        merged = RewardCalculator.merge_reward_config(base_config, scenario_overrides)

        comp = DriftPenalty(merged)
        comp.set_goal("AC001", {"lat": 39.25, "lon": 116.35})
        state = make_state(hdg=270.0)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        # drift ≈ π, penalty ≈ -0.5 * π
        assert result == pytest.approx(-0.5 * math.pi, abs=0.01)
