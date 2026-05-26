"""Tests for continuous action space support in parallel_env.

Verifies that the step() method correctly dispatches to translate_continuous()
when the action space is Box (continuous), and to translate() when MultiDiscrete.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from gymnasium import spaces

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.types import AircraftState, ConflictConfig, DiscreteAction, SpawnConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(acid="AC000", lat=40.0, lon=117.0, alt=35000.0, hdg=90.0, tas=450.0, vs=0.0):
    return AircraftState(
        id=acid, lat=lat, lon=lon, alt=alt, hdg=hdg, tas=tas, vs=vs,
    )


_MINIMAL_CONFIG = {
    "simulation": {"dt": 5.0, "action_frequency": 1, "max_episode_steps": 360},
    "aircraft": {"initial_count": 1, "spawn": {"altitude_range": [30000, 40000], "speed_range": [400, 500], "heading_range": [0, 360]}},
    "airspace": {"sectors": [{"id": "s0", "bounds": [[39.0, 116.0], [41.0, 118.0]], "capacity": 10}]},
    "rewards": {},
    "observation": {"max_other_aircraft": 10},
    "dynamic_entry": {"enabled": False},
}


class _DummyScenario(BaseScenario):
    """Minimal concrete scenario for testing."""

    name = "Dummy"
    _action_space_type = "discrete"

    def setup(self, rng, airspace_bounds):
        return ["AC000"]

    def get_spawn_config(self):
        return SpawnConfig(
            altitude_range=(30000, 40000),
            speed_range=(400, 500),
            heading_range=(0, 360),
        )

    def get_conflict_config(self):
        return ConflictConfig(
            nmac_horizontal_nm=5.0,
            nmac_vertical_ft=1000.0,
            warning_horizontal_nm=10.0,
            warning_vertical_ft=2000.0,
        )

    def get_waypoint(self, agent_id: str):
        return {"lat": 40.5, "lon": 117.5}


def _make_env(action_space_type="discrete"):
    """Create a BlueSkyMARLEnv with mocked dependencies."""
    from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv

    scenario = _DummyScenario()
    scenario._action_space_type = action_space_type

    wrapper = MagicMock()
    # _get_all_aircraft_states expects dicts, not AircraftState objects
    wrapper.get_all_aircraft_states.return_value = {
        "AC000": {"id": "AC000", "lat": 40.0, "lon": 117.0, "alt": 35000.0, "hdg": 90.0, "tas": 450.0, "vs": 0.0}
    }
    wrapper.is_aircraft_in_airspace.return_value = True
    wrapper.step_n.return_value = 10.0

    obs_manager = MagicMock()
    obs_manager.get_observations.return_value = ({"AC000": {}}, {"AC000": {}})

    translator = MagicMock(spec=ActionTranslator)
    translator.translate.return_value = ["HDG AC000 90"]
    translator.translate_continuous.return_value = ["HDG AC000 95"]

    reward_calc = MagicMock()
    reward_calc.compute.return_value = 0.0

    env = BlueSkyMARLEnv(
        config=_MINIMAL_CONFIG,
        wrapper=wrapper,
        observation_manager=obs_manager,
        action_translator=translator,
        reward_calculator=reward_calc,
        rewards_config={},
        scenario=scenario,
    )
    env._initialized = True
    env.agents = ["AC000"]
    env._prev_states = {"AC000": _make_state()}

    if action_space_type == "continuous":
        env._act_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    else:
        env._act_space = spaces.MultiDiscrete([5, 5, 5])

    return env


# ---------------------------------------------------------------------------
# Tests: step() dispatches to correct translator
# ---------------------------------------------------------------------------

class TestContinuousActionDispatch:
    """Verify step() calls translate_continuous for Box action space."""

    def test_continuous_action_calls_translate_continuous(self):
        """When action_space_type is 'continuous', step() uses translate_continuous."""
        env = _make_env("continuous")
        actions = {"AC000": np.array([0.5, 0.0, 0.0], dtype=np.float32)}
        env.step(actions)

        env._action_translator.translate_continuous.assert_called_once()
        env._action_translator.translate.assert_not_called()

    def test_discrete_action_calls_translate(self):
        """When action_space_type is 'discrete', step() uses translate (existing behavior)."""
        env = _make_env("discrete")
        actions = {"AC000": [2, 2, 2]}
        env.step(actions)

        env._action_translator.translate.assert_called_once()
        env._action_translator.translate_continuous.assert_not_called()

    def test_continuous_noop_produces_no_commands(self):
        """Continuous no-op [0,0,0] should produce no heading/alt/spd commands."""
        env = _make_env("continuous")
        # Replace mock translator with real one
        env._action_translator = ActionTranslator({"action": {}})

        actions = {"AC000": np.array([0.0, 0.0, 0.0], dtype=np.float32)}
        env.step(actions)

        env._wrapper.send_commands_batch.assert_called_once()
        sent_commands = env._wrapper.send_commands_batch.call_args[0][0]
        assert len(sent_commands) == 0


# ---------------------------------------------------------------------------
# Tests: SmoothnessPenalty with continuous actions
# ---------------------------------------------------------------------------

class TestSmoothnessPenaltyContinuous:
    """SmoothnessPenalty should work with continuous action arrays."""

    def test_continuous_nonzero_action_penalized(self):
        from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty

        config = {"components": {"smoothness": {"action_penalty": -0.1}}}
        comp = SmoothnessPenalty(config)
        state = _make_state()
        action = np.array([0.5, 0.0, 0.0])

        result = comp.compute("AC000", state, action, state, {}, step_count=0)
        assert result == -0.1

    def test_continuous_zero_action_no_penalty(self):
        from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty

        config = {"components": {"smoothness": {"action_penalty": -0.1}}}
        comp = SmoothnessPenalty(config)
        state = _make_state()
        action = np.array([0.0, 0.0, 0.0])

        result = comp.compute("AC000", state, action, state, {}, step_count=0)
        assert result == 0.0

    def test_discrete_action_still_works(self):
        from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty

        config = {"components": {"smoothness": {"action_penalty": -0.1}}}
        comp = SmoothnessPenalty(config)
        state = _make_state()

        # No-op discrete action
        action_noop = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)
        assert comp.compute("AC000", state, action_noop, state, {}, step_count=0) == 0.0

        # Non-zero discrete action
        action_move = DiscreteAction(heading_idx=0, altitude_idx=2, speed_idx=2)
        assert comp.compute("AC000", state, action_move, state, {}, step_count=0) == -0.1


# ---------------------------------------------------------------------------
# Tests: BaseScenario setter fix
# ---------------------------------------------------------------------------

class TestBaseScenarioSetter:
    """BaseScenario.action_space_type setter should not have dead code."""

    def test_setter_stores_value(self):
        scenario = _DummyScenario()
        scenario.action_space_type = "continuous"
        assert scenario.action_space_type == "continuous"

    def test_setter_default_is_discrete(self):
        scenario = _DummyScenario()
        assert scenario.action_space_type == "discrete"
