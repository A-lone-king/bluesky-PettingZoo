"""Backward compatibility integration tests (G-V02).

Verify that V2.0 changes do not break V1.0 functionality:
- No-scenario env works as before
- Observation space is compatible
- Reward calculation unaffected
- PettingZoo API compliance
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from gymnasium import spaces

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction

from tests.helpers.fake_wrapper import FakeBlueSkyWrapper
from tests.helpers.env_factory import make_config as _make_config
from tests.helpers.env_factory import write_rewards_yaml as _write_rewards_yaml
from tests.helpers.env_factory import make_env as _make_env


# ---------------------------------------------------------------------------
# Fake BlueSkyWrapper
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------





class TestV1EnvStillWorks:
    """V1.0 env without scenario should work as before."""

    def test_v1_env_still_works(self, tmp_path: Path) -> None:
        """Reset and step cycle completes without error."""
        config = _make_config(initial_count=3, max_steps=10)
        env = _make_env(tmp_path, config)

        obs, infos = env.reset(seed=42)
        assert len(env.agents) == 3

        for _ in range(5):
            actions = {a: [2, 2, 2] for a in env.agents}
            if not actions:
                break
            obs, rewards, terminations, truncations, infos = env.step(actions)

        env.close()


class TestV1ObservationCompatible:
    """Observation space should be compatible with V1.0 code."""

    def test_v1_observation_compatible(self, tmp_path: Path) -> None:
        """Observation space is Dict with expected keys."""
        config = _make_config(initial_count=3)
        env = _make_env(tmp_path, config)
        env.reset(seed=42)

        agent = env.agents[0]
        obs_space = env.observation_space(agent)

        assert isinstance(obs_space, spaces.Dict)
        assert "self_state" in obs_space.spaces
        assert "other_aircraft" in obs_space.spaces
        assert "goal" in obs_space.spaces

        # self_state should be Box with reasonable shape
        self_space = obs_space.spaces["self_state"]
        assert isinstance(self_space, spaces.Box)
        assert len(self_space.shape) == 1


class TestV1RewardCompatible:
    """Reward calculation should not be affected by V2.0 changes."""

    def test_v1_reward_compatible(self, tmp_path: Path) -> None:
        """Reward components produce finite values."""
        config = _make_config(initial_count=2, max_steps=10)
        env = _make_env(tmp_path, config)
        env.reset(seed=42)

        actions = {a: [2, 2, 2] for a in env.agents}
        _, rewards, _, _, _ = env.step(actions)

        for agent_id, r in rewards.items():
            assert np.isfinite(r), f"Reward for {agent_id} is not finite: {r}"


class TestV1APICompliance:
    """PettingZoo ParallelEnv API should still be satisfied."""

    def test_v1_api_compliance(self, tmp_path: Path) -> None:
        """Env satisfies PettingZoo ParallelEnv interface."""
        config = _make_config(initial_count=3, max_steps=10)
        env = _make_env(tmp_path, config)

        # Required attributes
        assert hasattr(env, "agents")
        assert hasattr(env, "possible_agents")
        assert isinstance(env.agents, list)

        # reset returns (obs, infos)
        result = env.reset(seed=42)
        assert len(result) == 2
        obs, infos = result
        assert isinstance(obs, dict)
        assert isinstance(infos, dict)

        # step returns 5-tuple
        actions = {a: [2, 2, 2] for a in env.agents}
        step_result = env.step(actions)
        assert len(step_result) == 5
        obs, rewards, terminations, truncations, infos = step_result
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terminations, dict)
        assert isinstance(truncations, dict)
        assert isinstance(infos, dict)

        # observation_space and action_space exist
        agent = env.possible_agents[0]
        assert env.observation_space(agent) is not None
        assert env.action_space(agent) is not None

        env.close()
