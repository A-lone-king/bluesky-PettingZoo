"""Tests for MergeScenario (T-V16).

Approach merge: 1 controllable aircraft + 19 background traffic.
Background traffic follows preset routes (uncontrollable).
The controllable aircraft must find a safe gap to merge.
Conflict distance: 4 NM (stricter than cruise).
Observable neighbors limited to 5.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.merge import MergeScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from bluesky_pettingzoo.utils.geometry import haversine_distance
from bluesky_pettingzoo.utils.types import AircraftState, ConflictConfig, SpawnConfig

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from tests.helpers.env_factory import make_config as _make_config
from tests.helpers.env_factory import write_rewards_yaml as _write_rewards_yaml


# ---------------------------------------------------------------------------
# Fake BlueSkyWrapper
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------




def _make_env_with_scenario(
    env_config: dict[str, Any],
    scenario: MergeScenario,
) -> BlueSkyMARLEnv:
    """Create a BlueSkyMARLEnv with a MergeScenario."""
    wrapper = BlueSkyWrapper(env_config)
    obs_manager = ObservationManager(env_config)
    action_translator = ActionTranslator(env_config)

    rewards_path = env_config["_rewards_yaml"]
    with open(rewards_path, encoding="utf-8") as f:
        rewards_cfg = yaml.safe_load(f)
    merged = {**env_config, **rewards_cfg}

    calc = RewardCalculator()
    calc.register(ConflictPenalty(merged), weight=1.0)
    calc.register(SmoothnessPenalty(merged), weight=0.5)
    eff = EfficiencyReward(merged)
    calc.register(eff, weight=0.3)

    return BlueSkyMARLEnv(
        config=env_config,
        wrapper=wrapper,
        observation_manager=obs_manager,
        action_translator=action_translator,
        reward_calculator=calc,
        rewards_config=rewards_cfg,
        scenario=scenario,
    )


# ===========================================================================
# T-V16 tests
# ===========================================================================


class TestMergeSetup:
    """Scenario initialization should succeed with 20 aircraft."""

    def test_merge_setup(self) -> None:
        """setup() returns 20 agent IDs (1 controllable + 19 background)."""
        scenario = MergeScenario(num_aircraft=20, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 40.0, "lon_min": 116.0, "lon_max": 117.0}
        agents = scenario.setup(rng, bounds)

        assert len(agents) == 20
        assert all(isinstance(a, str) for a in agents)


class TestMergeBackgroundTrafficUncontrollable:
    """Background traffic should be marked as uncontrollable."""

    def test_merge_background_traffic_uncontrollable(self) -> None:
        """Only the first agent is controllable; rest are background."""
        scenario = MergeScenario(num_aircraft=20, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 40.0, "lon_min": 116.0, "lon_max": 117.0}
        agents = scenario.setup(rng, bounds)

        controllable = scenario.get_controllable_agents()
        assert len(controllable) == 1
        assert controllable[0] == agents[0]

        background = scenario.get_background_agents()
        assert len(background) == 19


class TestMergeConflictDistance4nm:
    """Merge scenario should use 4 NM conflict distance."""

    def test_merge_conflict_distance_4nm(self) -> None:
        """Conflict config has 4 NM NMAC horizontal threshold."""
        scenario = MergeScenario(num_aircraft=20, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 40.0, "lon_min": 116.0, "lon_max": 117.0}
        scenario.setup(rng, bounds)

        conflict_cfg = scenario.get_conflict_config()
        assert conflict_cfg.nmac_horizontal_nm == 4.0


class TestMergeObservableNeighbors:
    """Observable neighbors should be limited to 5."""

    def test_merge_observable_neighbors(self, tmp_path: Path) -> None:
        """Observation config limits observable aircraft to 5."""
        _write_rewards_yaml(tmp_path)
        config = _make_config()
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = MergeScenario(num_aircraft=20, seed=42)
        env = _make_env_with_scenario(config, scenario)
        env.reset(seed=42)

        # The observation space should limit to 5 observable aircraft
        obs_space = env.observation_space(env.agents[0])
        if "other_aircraft" in obs_space.spaces:
            other_shape = obs_space.spaces["other_aircraft"].shape
            assert other_shape[0] == 5  # max_observable_aircraft


class TestMergeArrivalAtFAF:
    """Aircraft reaching FAF should trigger termination."""

    def test_merge_arrival_at_faf(self, tmp_path: Path) -> None:
        """Controllable aircraft at FAF terminates."""
        _write_rewards_yaml(tmp_path)
        config = _make_config()
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = MergeScenario(num_aircraft=20, seed=42)
        env = _make_env_with_scenario(config, scenario)
        env.reset(seed=42)

        agents = list(env.agents)
        wrapper = env._wrapper
        controllable = scenario.get_controllable_agents()
        acid = controllable[0]

        # Place aircraft at its FAF waypoint
        faf = scenario.get_waypoint(acid)
        wrapper.set_aircraft_state(acid, lat=faf["lat"], lon=faf["lon"], alt=faf["alt"])

        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, terminations, _, _ = env.step(actions)

        assert terminations[acid] is True


class TestMergeFullEpisode:
    """Full episode with MergeScenario should run without errors."""

    def test_merge_full_episode(self, tmp_path: Path) -> None:
        """Run a complete episode with the scenario."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(max_steps=20)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = MergeScenario(num_aircraft=20, seed=42)
        env = _make_env_with_scenario(config, scenario)
        obs, infos = env.reset(seed=42)

        assert len(env.agents) == 20

        total_reward = 0.0
        for step in range(20):
            actions = {a: [2, 2, 2] for a in env.agents}
            if not actions:
                break
            obs, rewards, terminations, truncations, infos = env.step(actions)
            total_reward += sum(rewards.values())

            if not env.agents:
                break

        assert np.isfinite(total_reward)
