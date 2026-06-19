"""Tests for distance reward in EfficiencyReward component."""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.utils.types import AircraftState


def _make_state(lat: float, lon: float, alt: float = 30000.0) -> AircraftState:
    """Create a minimal AircraftState."""
    return AircraftState(id="AC000", lat=lat, lon=lon, alt=alt, hdg=90.0, tas=450.0, vs=0.0)


class TestDistanceReward:
    """Distance reward should provide progressive reward for approaching goal."""

    def test_distance_reward_disabled_by_default(self) -> None:
        """When distance_reward_scale=0, no distance reward is given."""
        config = {"components": {"efficiency": {"distance_reward_scale": 0.0}}}
        eff = EfficiencyReward(config)
        eff.set_goal("AC000", lat=40.0, lon=117.0, initial_lat=39.0, initial_lon=116.0)

        state = _make_state(lat=39.5, lon=116.5)
        reward = eff.compute("AC000", state, [2, 2, 2], state, {})

        # Only step_penalty, no distance reward
        assert reward == pytest.approx(-0.0)

    def test_distance_reward_positive_when_approaching(self) -> None:
        """Agent closer to goal should get higher distance reward."""
        config = {
            "components": {
                "efficiency": {"distance_reward_scale": 1.0, "distance_threshold_nm": 100.0}
            }
        }
        eff = EfficiencyReward(config)
        eff.set_goal("AC000", lat=40.0, lon=117.0, initial_lat=39.0, initial_lon=116.0)

        # Far from goal
        state_far = _make_state(lat=39.1, lon=116.1)
        reward_far = eff.compute("AC000", state_far, [2, 2, 2], state_far, {})

        # Closer to goal
        state_near = _make_state(lat=39.9, lon=116.9)
        reward_near = eff.compute("AC000", state_near, [2, 2, 2], state_near, {})

        assert reward_near > reward_far

    def test_distance_reward_at_goal(self) -> None:
        """Agent at goal should get maximum distance reward."""
        config = {
            "components": {
                "efficiency": {"distance_reward_scale": 1.0, "distance_threshold_nm": 100.0}
            }
        }
        eff = EfficiencyReward(config)
        eff.set_goal("AC000", lat=40.0, lon=117.0, initial_lat=39.0, initial_lon=116.0)

        state_at_goal = _make_state(lat=40.0, lon=117.0)
        reward = eff.compute("AC000", state_at_goal, [2, 2, 2], state_at_goal, {})

        # distance_reward (1.0) + arrival_reward (1.0) = 2.0
        assert reward == pytest.approx(2.0)

    def test_distance_reward_respects_threshold(self) -> None:
        """Agent beyond distance_threshold should not get distance reward."""
        config = {
            "components": {
                "efficiency": {"distance_reward_scale": 1.0, "distance_threshold_nm": 10.0}
            }
        }
        eff = EfficiencyReward(config)
        eff.set_goal("AC000", lat=40.0, lon=117.0, initial_lat=39.0, initial_lon=116.0)

        # Far beyond threshold (~80nm)
        state_far = _make_state(lat=39.1, lon=116.1)
        reward = eff.compute("AC000", state_far, [2, 2, 2], state_far, {})

        # No distance reward beyond threshold
        assert reward == pytest.approx(0.0)

    def test_distance_reward_without_initial_distance(self) -> None:
        """Agent without initial distance should not get distance reward."""
        config = {"components": {"efficiency": {"distance_reward_scale": 1.0}}}
        eff = EfficiencyReward(config)
        eff.set_goal("AC000", lat=40.0, lon=117.0)  # No initial_lat/lon

        state = _make_state(lat=39.5, lon=116.5)
        reward = eff.compute("AC000", state, [2, 2, 2], state, {})

        assert reward == pytest.approx(0.0)

    def test_distance_reward_combined_with_arrival(self) -> None:
        """Distance reward and arrival reward can both apply."""
        config = {
            "components": {
                "efficiency": {
                    "distance_reward_scale": 1.0,
                    "arrival_reward": 10.0,
                    "arrival_threshold_nm": 2.0,
                }
            }
        }
        eff = EfficiencyReward(config)
        eff.set_goal("AC000", lat=40.0, lon=117.0, initial_lat=39.0, initial_lon=116.0)

        # At goal (within arrival threshold)
        state = _make_state(lat=40.0, lon=117.0)
        reward = eff.compute("AC000", state, [2, 2, 2], state, {})

        # distance_reward (1.0) + arrival_reward (10.0) = 11.0
        assert reward == pytest.approx(11.0)
