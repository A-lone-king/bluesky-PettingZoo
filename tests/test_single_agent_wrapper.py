"""Tests for SingleAgentGymWrapper."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from gymnasium import spaces

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from bluesky_pettingzoo.wrappers.single_agent import SingleAgentGymWrapper

from tests.helpers.fake_wrapper import FakeBlueSkyWrapper
from tests.helpers.env_factory import make_config, write_rewards_yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wrapped_env(
    tmp_path,
    num_aircraft: int = 3,
    max_steps: int = 50,
    ego_agent: str = "AC000",
) -> SingleAgentGymWrapper:
    """Create a SingleAgentGymWrapper around a WaypointNav env."""
    config = make_config(initial_count=num_aircraft, max_steps=max_steps)
    rewards_path = write_rewards_yaml(tmp_path)
    config["_rewards_yaml"] = str(rewards_path)

    with open(rewards_path, encoding="utf-8") as f:
        rewards_cfg = yaml.safe_load(f)
    merged = {**config, **rewards_cfg}

    wrapper = FakeBlueSkyWrapper(config)
    obs_manager = ObservationManager(config)
    action_translator = ActionTranslator(config)
    calc = RewardCalculator()
    calc.register(ConflictPenalty(merged), weight=1.0)
    calc.register(SmoothnessPenalty(merged), weight=0.5)
    calc.register(EfficiencyReward(merged), weight=0.3)

    env = BlueSkyMARLEnv(
        config=config,
        wrapper=wrapper,
        observation_manager=obs_manager,
        action_translator=action_translator,
        reward_calculator=calc,
        rewards_config=rewards_cfg,
        scenario=WaypointNavScenario(num_aircraft=num_aircraft, seed=42),
    )
    return SingleAgentGymWrapper(env, ego_agent=ego_agent)


# ===========================================================================
# Tests
# ===========================================================================


class TestSingleAgentGymWrapperSpaces:
    def test_observation_space_is_dict(self, tmp_path) -> None:
        wrapper = _make_wrapped_env(tmp_path)
        assert isinstance(wrapper.observation_space, spaces.Dict)
        assert "self_state" in wrapper.observation_space.spaces
        assert "goal" in wrapper.observation_space.spaces

    def test_action_space_is_multidiscrete(self, tmp_path) -> None:
        wrapper = _make_wrapped_env(tmp_path)
        assert isinstance(wrapper.action_space, spaces.MultiDiscrete)
        assert list(wrapper.action_space.nvec) == [5, 5, 5]


class TestSingleAgentGymWrapperReset:
    def test_reset_returns_obs_and_info(self, tmp_path) -> None:
        wrapper = _make_wrapped_env(tmp_path)
        obs, info = wrapper.reset(seed=42)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_reset_observation_in_space(self, tmp_path) -> None:
        wrapper = _make_wrapped_env(tmp_path)
        obs, _ = wrapper.reset(seed=42)
        assert wrapper.observation_space.contains(obs)

    def test_reset_ego_agent_active(self, tmp_path) -> None:
        wrapper = _make_wrapped_env(tmp_path, ego_agent="AC000")
        wrapper.reset(seed=42)
        assert "AC000" in wrapper._env.agents


class TestSingleAgentGymWrapperStep:
    def test_step_returns_five_tuple(self, tmp_path) -> None:
        wrapper = _make_wrapped_env(tmp_path)
        wrapper.reset(seed=42)
        result = wrapper.step(np.array([2, 2, 2]))
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_observation_in_space(self, tmp_path) -> None:
        wrapper = _make_wrapped_env(tmp_path)
        wrapper.reset(seed=42)
        obs, _, _, _, _ = wrapper.step(np.array([2, 2, 2]))
        assert wrapper.observation_space.contains(obs)

    def test_step_reward_is_finite(self, tmp_path) -> None:
        wrapper = _make_wrapped_env(tmp_path)
        wrapper.reset(seed=42)
        _, reward, _, _, _ = wrapper.step(np.array([2, 2, 2]))
        assert np.isfinite(reward)


class TestSingleAgentGymWrapperTermination:
    def test_truncation_at_max_steps(self, tmp_path) -> None:
        wrapper = _make_wrapped_env(tmp_path, max_steps=3)
        wrapper.reset(seed=42)
        for _ in range(5):
            _, _, terminated, truncated, _ = wrapper.step(np.array([2, 2, 2]))
            if terminated or truncated:
                break
        assert truncated or terminated


class TestSingleAgentGymWrapperFullEpisode:
    def test_full_episode_with_noop(self, tmp_path) -> None:
        wrapper = _make_wrapped_env(tmp_path, max_steps=20)
        wrapper.reset(seed=42)
        total_reward = 0.0
        for _ in range(25):
            _, reward, terminated, truncated, _ = wrapper.step(np.array([2, 2, 2]))
            total_reward += reward
            if terminated or truncated:
                break
        assert np.isfinite(total_reward)

    def test_full_episode_with_random_actions(self, tmp_path) -> None:
        wrapper = _make_wrapped_env(tmp_path, max_steps=20)
        wrapper.reset(seed=42)
        rng = np.random.RandomState(42)
        total_reward = 0.0
        for _ in range(25):
            action = wrapper.action_space.sample()
            _, reward, terminated, truncated, _ = wrapper.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        assert np.isfinite(total_reward)


class TestSingleAgentGymWrapperGymnasiumCompliance:
    def test_is_gymnasium_env(self, tmp_path) -> None:
        import gymnasium
        wrapper = _make_wrapped_env(tmp_path)
        assert isinstance(wrapper, gymnasium.Env)

    def test_has_required_attrs(self, tmp_path) -> None:
        wrapper = _make_wrapped_env(tmp_path)
        assert hasattr(wrapper, "observation_space")
        assert hasattr(wrapper, "action_space")
        assert hasattr(wrapper, "reset")
        assert hasattr(wrapper, "step")
        assert hasattr(wrapper, "close")
