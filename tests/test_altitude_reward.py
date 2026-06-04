"""Tests for AltitudeReward — regime-dependent altitude penalty.

Three regimes inspired by bluesky-gym VerticalCR/Descent:
- Enroute: gentle penalty proportional to altitude error
- Near runway: steep penalty for remaining altitude
- Crash: fixed penalty when altitude <= 0
"""

from __future__ import annotations

from bluesky_pettingzoo.utils.types import AircraftState


def _make_state(acid="AC000", lat=40.0, lon=117.0, alt=35000.0, hdg=90.0, tas=450.0, vs=0.0):
    return AircraftState(id=acid, lat=lat, lon=lon, alt=alt, hdg=hdg, tas=tas, vs=vs)


class TestAltitudeRewardEnroute:
    """Enroute regime: gentle penalty for altitude error."""

    def test_no_altitude_error(self):
        from bluesky_pettingzoo.rewards.components.altitude_reward import AltitudeReward

        config = {
            "components": {
                "altitude_reward": {
                    "enroute_scale": 5.0 / 3000,
                    "runway_scale": 50.0 / 3000,
                    "runway_threshold_nm": 5.0,
                    "crash_penalty": -100.0,
                }
            }
        }
        comp = AltitudeReward(config)
        comp.set_goal("AC000", 40.5, 117.5, target_alt=35000.0)

        state = _make_state(alt=35000.0)
        result = comp.compute("AC000", state, [0, 0, 0], state, {}, step_count=0)
        assert result == 0.0

    def test_altitude_error_gives_penalty(self):
        from bluesky_pettingzoo.rewards.components.altitude_reward import AltitudeReward

        config = {
            "components": {
                "altitude_reward": {
                    "enroute_scale": 5.0 / 3000,
                    "runway_scale": 50.0 / 3000,
                    "runway_threshold_nm": 5.0,
                    "crash_penalty": -100.0,
                }
            }
        }
        comp = AltitudeReward(config)
        comp.set_goal("AC000", 40.5, 117.5, target_alt=30000.0)

        state = _make_state(alt=35000.0)
        result = comp.compute("AC000", state, [0, 0, 0], state, {}, step_count=0)
        # 5000 ft error * (-5/3000) ≈ -8.333
        assert result < 0
        assert abs(result - (-5000.0 * 5.0 / 3000)) < 0.01

    def test_altitude_error_symmetric(self):
        from bluesky_pettingzoo.rewards.components.altitude_reward import AltitudeReward

        config = {
            "components": {
                "altitude_reward": {
                    "enroute_scale": 5.0 / 3000,
                    "runway_scale": 50.0 / 3000,
                    "runway_threshold_nm": 5.0,
                    "crash_penalty": -100.0,
                }
            }
        }
        comp = AltitudeReward(config)
        comp.set_goal("AC000", 40.5, 117.5, target_alt=30000.0)

        state_high = _make_state(alt=35000.0)
        state_low = _make_state(alt=25000.0)
        result_high = comp.compute("AC000", state_high, [0, 0, 0], state_high, {}, step_count=0)
        result_low = comp.compute("AC000", state_low, [0, 0, 0], state_low, {}, step_count=0)
        assert abs(result_high - result_low) < 0.01


class TestAltitudeRewardNearRunway:
    """Near runway regime: steep penalty for remaining altitude."""

    def test_near_runway_steeper_penalty(self):
        from bluesky_pettingzoo.rewards.components.altitude_reward import AltitudeReward

        config = {
            "components": {
                "altitude_reward": {
                    "enroute_scale": 5.0 / 3000,
                    "runway_scale": 50.0 / 3000,
                    "runway_threshold_nm": 5.0,
                    "crash_penalty": -100.0,
                }
            }
        }
        comp = AltitudeReward(config)
        # Goal is very close (1 NM away) at altitude 3000
        comp.set_goal("AC000", 40.01, 117.01, target_alt=3000.0)

        # Aircraft at 10000 ft, 1 NM from goal
        state = _make_state(alt=10000.0)
        result = comp.compute("AC000", state, [0, 0, 0], state, {}, step_count=0)
        # remaining_alt = 10000 - 3000 = 7000
        # 7000 * (-50/3000) ≈ -116.67
        assert result < -100

    def test_far_from_runway_uses_enroute(self):
        from bluesky_pettingzoo.rewards.components.altitude_reward import AltitudeReward

        config = {
            "components": {
                "altitude_reward": {
                    "enroute_scale": 5.0 / 3000,
                    "runway_scale": 50.0 / 3000,
                    "runway_threshold_nm": 5.0,
                    "crash_penalty": -100.0,
                }
            }
        }
        comp = AltitudeReward(config)
        # Goal is far away (10 NM) at altitude 3000
        comp.set_goal("AC000", 41.0, 118.0, target_alt=3000.0)

        state = _make_state(alt=10000.0)
        result = comp.compute("AC000", state, [0, 0, 0], state, {}, step_count=0)
        # Should use enroute scale: 7000 * (-5/3000) ≈ -11.67
        assert -15 < result < -5


class TestAltitudeRewardCrash:
    """Crash regime: fixed penalty when altitude <= 0."""

    def test_crash_penalty(self):
        from bluesky_pettingzoo.rewards.components.altitude_reward import AltitudeReward

        config = {
            "components": {
                "altitude_reward": {
                    "enroute_scale": 5.0 / 3000,
                    "runway_scale": 50.0 / 3000,
                    "runway_threshold_nm": 5.0,
                    "crash_penalty": -100.0,
                }
            }
        }
        comp = AltitudeReward(config)
        comp.set_goal("AC000", 40.5, 117.5, target_alt=3000.0)

        state = _make_state(alt=0.0)
        result = comp.compute("AC000", state, [0, 0, 0], state, {}, step_count=0)
        assert result == -100.0

    def test_negative_altitude_crash(self):
        from bluesky_pettingzoo.rewards.components.altitude_reward import AltitudeReward

        config = {
            "components": {
                "altitude_reward": {
                    "enroute_scale": 5.0 / 3000,
                    "runway_scale": 50.0 / 3000,
                    "runway_threshold_nm": 5.0,
                    "crash_penalty": -100.0,
                }
            }
        }
        comp = AltitudeReward(config)
        comp.set_goal("AC000", 40.5, 117.5, target_alt=3000.0)

        state = _make_state(alt=-100.0)
        result = comp.compute("AC000", state, [0, 0, 0], state, {}, step_count=0)
        assert result == -100.0


class TestAltitudeRewardNoGoal:
    """Without a goal set, reward should be 0."""

    def test_no_goal_returns_zero(self):
        from bluesky_pettingzoo.rewards.components.altitude_reward import AltitudeReward

        config = {
            "components": {
                "altitude_reward": {
                    "enroute_scale": 5.0 / 3000,
                    "runway_scale": 50.0 / 3000,
                    "runway_threshold_nm": 5.0,
                    "crash_penalty": -100.0,
                }
            }
        }
        comp = AltitudeReward(config)

        state = _make_state(alt=35000.0)
        result = comp.compute("AC000", state, [0, 0, 0], state, {}, step_count=0)
        assert result == 0.0
