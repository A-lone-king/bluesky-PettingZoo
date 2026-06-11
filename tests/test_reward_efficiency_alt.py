"""Tests for efficiency reward altitude deviation (reward-002)."""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from tests.helpers.state_factory import make_action, make_state


class TestAltDeviation:
    """Test altitude deviation penalty."""

    def test_alt_deviation_penalty(self, rewards_config: dict) -> None:
        """Altitude deviation of 1000ft → proportional penalty."""
        config = {
            **rewards_config,
            "components": {
                **rewards_config.get("components", {}),
                "efficiency": {
                    **rewards_config.get("components", {}).get("efficiency", {}),
                    "max_alt_deviation_ft": 5000.0,
                    "alt_deviation_penalty_scale": 2.0,
                },
            },
        }
        comp = EfficiencyReward(config)
        # Aircraft at 35000ft, goal at 36000ft → 1000ft deviation
        state = make_state(lat=39.25, lon=116.25, alt=35000.0)
        comp.set_goal("AC001", lat=39.25, lon=116.25, alt=36000.0)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        # alt_deviation_penalty = -(1000/5000)*2 = -0.4
        # step_penalty(-0.01) + deviation(0) + alt_penalty(-0.4) + arrival(+10)
        # arrival is triggered because horizontal distance = 0
        assert result == pytest.approx(9.59)

    def test_max_alt_deviation_penalty(self, rewards_config: dict) -> None:
        """Altitude deviation at MAX → max penalty."""
        config = {
            **rewards_config,
            "components": {
                **rewards_config.get("components", {}),
                "efficiency": {
                    **rewards_config.get("components", {}).get("efficiency", {}),
                    "max_alt_deviation_ft": 5000.0,
                    "alt_deviation_penalty_scale": 2.0,
                },
            },
        }
        comp = EfficiencyReward(config)
        # Aircraft at 30000ft, goal at 35000ft → 5000ft deviation
        state = make_state(lat=39.25, lon=116.25, alt=30000.0)
        comp.set_goal("AC001", lat=39.25, lon=116.25, alt=35000.0)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        # alt_deviation_penalty = -(5000/5000)*2 = -2.0
        # step_penalty(-0.01) + deviation(0) + alt_penalty(-2.0) + arrival(+10)
        # arrival is triggered because horizontal distance = 0
        assert result == pytest.approx(7.99)

    def test_no_alt_deviation(self, rewards_config: dict) -> None:
        """Aircraft at correct altitude → no altitude penalty."""
        config = {
            **rewards_config,
            "components": {
                **rewards_config.get("components", {}),
                "efficiency": {
                    **rewards_config.get("components", {}).get("efficiency", {}),
                    "max_alt_deviation_ft": 5000.0,
                    "alt_deviation_penalty_scale": 2.0,
                },
            },
        }
        comp = EfficiencyReward(config)
        state = make_state(lat=39.25, lon=116.25, alt=35000.0)
        comp.set_goal("AC001", lat=39.25, lon=116.25, alt=35000.0)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        # No alt penalty, at goal → arrival reward
        # total = -0.01 + 0 + 0 + 10 = 9.99
        assert result == pytest.approx(9.99)


class TestNoAltGoal:
    """Test behavior when no altitude goal is set."""

    def test_no_alt_goal_ignores_alt_penalty(self, rewards_config: dict) -> None:
        """When set_goal() has no alt parameter, altitude penalty is 0."""
        config = {
            **rewards_config,
            "components": {
                **rewards_config.get("components", {}),
                "efficiency": {
                    **rewards_config.get("components", {}).get("efficiency", {}),
                    "max_alt_deviation_ft": 5000.0,
                    "alt_deviation_penalty_scale": 2.0,
                },
            },
        }
        comp = EfficiencyReward(config)
        state = make_state(lat=39.25, lon=116.25, alt=35000.0)
        # Only set lat/lon goal, no alt
        comp.set_goal("AC001", lat=39.25, lon=116.25)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        # No alt penalty, at goal → arrival reward
        # total = -0.01 + 0 + 0 + 10 = 9.99
        assert result == pytest.approx(9.99)


class TestCombinedDeviation:
    """Test combined horizontal and altitude deviation."""

    def test_combined_deviation(self, rewards_config: dict) -> None:
        """Both horizontal and altitude deviations penalized."""
        config = {
            **rewards_config,
            "components": {
                **rewards_config.get("components", {}),
                "efficiency": {
                    **rewards_config.get("components", {}).get("efficiency", {}),
                    "max_alt_deviation_ft": 5000.0,
                    "alt_deviation_penalty_scale": 2.0,
                },
            },
        }
        comp = EfficiencyReward(config)
        # Aircraft at (39.25, 116.25, 35000)
        state = make_state(lat=39.25, lon=116.25, alt=35000.0)
        # Goal at (39.4165, 116.25, 36000) → 10NM + 1000ft
        comp.set_goal("AC001", lat=39.4165543515, lon=116.25, alt=36000.0)
        action = make_action()

        result = comp.compute("AC001", state, action, state, {"AC001": state})

        # horizontal: -(10/50)*5 = -1.0
        # altitude: -(1000/5000)*2 = -0.4
        # total = -0.01 + (-1.0) + (-0.4) = -1.41
        assert result == pytest.approx(-1.41)
