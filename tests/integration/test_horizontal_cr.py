"""Tests for HorizontalCRScenario (T-V09).

Horizontal conflict resolution: multiple aircraft cruise at the same altitude,
use heading maneuvers to avoid conflicts, and terminate upon reaching waypoints.
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
from bluesky_pettingzoo.envs.scenarios.horizontal_cr import HorizontalCRScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from bluesky_pettingzoo.utils.geometry import haversine_distance
from bluesky_pettingzoo.utils.types import AircraftState, ConflictConfig, SpawnConfig

from tests.helpers.fake_wrapper import FakeBlueSkyWrapper
from tests.helpers.env_factory import make_config as _make_config
from tests.helpers.env_factory import write_rewards_yaml as _write_rewards_yaml


# ---------------------------------------------------------------------------
# Fake BlueSkyWrapper (same pattern as test_env.py)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------




def _make_env_with_scenario(
    env_config: dict[str, Any],
    scenario: HorizontalCRScenario,
) -> BlueSkyMARLEnv:
    """Create a BlueSkyMARLEnv with a HorizontalCRScenario."""
    wrapper = FakeBlueSkyWrapper(env_config)
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
# T-V09 tests
# ===========================================================================


class TestHorizontalCRSetup:
    """Scenario initialization should succeed with correct aircraft count."""

    def test_horizontal_cr_setup(self, tmp_path: Path) -> None:
        """setup() returns the configured number of agent IDs."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=4)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = HorizontalCRScenario(num_aircraft=4, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        agents = scenario.setup(rng, bounds)

        assert len(agents) == 4
        assert all(isinstance(a, str) for a in agents)
        # All agent IDs should be unique
        assert len(set(agents)) == 4


class TestHorizontalCRConflictDetected:
    """Head-on aircraft should be detected as conflicting."""

    def test_horizontal_cr_conflict_detected(self, tmp_path: Path) -> None:
        """Two aircraft on head-on collision course are detected as conflict."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=2)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = HorizontalCRScenario(num_aircraft=2, seed=42)
        env = _make_env_with_scenario(config, scenario)
        env.reset(seed=42)

        # Manually position aircraft on head-on course (same altitude)
        wrapper = env._wrapper
        agents = list(env.agents)
        # Place them 8 NM apart, same altitude, heading toward each other
        wrapper._aircraft[agents[0]].update({"lat": 39.25, "lon": 116.2, "alt": 35000, "hdg": 90.0})
        wrapper._aircraft[agents[1]].update({"lat": 39.25, "lon": 116.35, "alt": 35000, "hdg": 270.0})

        dist = haversine_distance(39.25, 116.2, 39.25, 116.35)
        assert dist < 10.0  # Within warning distance

        # Step and check that conflict is detected in infos
        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, _, _, infos = env.step(actions)
        for agent_id in infos:
            ts = infos[agent_id].get("textual_state", {})
            assert ts.get("conflict_status") in ("warning", "nmac")


class TestHorizontalCRNoConflictSafe:
    """Parallel-flying aircraft far apart should have no conflict."""

    def test_horizontal_cr_no_conflict_safe(self, tmp_path: Path) -> None:
        """Two aircraft far apart and parallel report safe conflict status."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=2)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = HorizontalCRScenario(num_aircraft=2, seed=42)
        env = _make_env_with_scenario(config, scenario)
        env.reset(seed=42)

        # Position aircraft far apart, parallel flight
        wrapper = env._wrapper
        agents = list(env.agents)
        wrapper._aircraft[agents[0]].update({"lat": 39.1, "lon": 116.1, "alt": 35000, "hdg": 90.0})
        wrapper._aircraft[agents[1]].update({"lat": 39.4, "lon": 116.4, "alt": 33000, "hdg": 90.0})

        dist = haversine_distance(39.1, 116.1, 39.4, 116.4)
        assert dist > 10.0  # Beyond warning distance

        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, _, _, infos = env.step(actions)
        for agent_id in infos:
            ts = infos[agent_id].get("textual_state", {})
            assert ts.get("conflict_status") == "safe"


class TestHorizontalCRArrivalTermination:
    """Aircraft reaching its waypoint should trigger termination."""

    def test_horizontal_cr_arrival_termination(self, tmp_path: Path) -> None:
        """Aircraft within arrival_threshold of goal is terminated."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=2)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = HorizontalCRScenario(num_aircraft=2, seed=42)
        env = _make_env_with_scenario(config, scenario)
        env.reset(seed=42)

        agents = list(env.agents)
        wrapper = env._wrapper

        # Place first aircraft very close to its waypoint (within 2 NM)
        wp = scenario.get_waypoint(agents[0])
        wrapper._aircraft[agents[0]].update({
            "lat": wp["lat"],
            "lon": wp["lon"],
            "alt": wp["alt"],
            "hdg": 90.0,
        })

        # Place second aircraft far from its waypoint
        wp2 = scenario.get_waypoint(agents[1])
        wrapper._aircraft[agents[1]].update({
            "lat": 39.1,
            "lon": 116.1,
            "alt": 33000,
            "hdg": 90.0,
        })

        initial_agents = set(env.agents)
        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, terminations, _, _ = env.step(actions)

        # First aircraft should be terminated (arrived at waypoint)
        assert terminations[agents[0]] is True
        assert agents[0] not in env.agents

        # Second aircraft should still be active
        assert terminations[agents[1]] is False
        assert agents[1] in env.agents


class TestHorizontalCRActionHeadingOnly:
    """HorizontalCR scenario should restrict action space to heading only."""

    def test_horizontal_cr_action_heading_only(self, tmp_path: Path) -> None:
        """Scenario reports heading-only action constraints."""
        scenario = HorizontalCRScenario(num_aircraft=3, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        scenario.setup(rng, bounds)

        # The scenario should indicate that only heading actions are valid
        assert scenario.action_dimensions == [0]  # heading index only


class TestHorizontalCRFullEpisode:
    """Full episode with HorizontalCRScenario should run without errors."""

    def test_horizontal_cr_full_episode(self, tmp_path: Path) -> None:
        """Run a complete episode with the scenario."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=3, max_steps=20)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = HorizontalCRScenario(num_aircraft=3, seed=42)
        env = _make_env_with_scenario(config, scenario)
        obs, infos = env.reset(seed=42)

        assert len(env.agents) == 3

        total_reward = 0.0
        for step in range(20):
            actions = {a: [2, 2, 2] for a in env.agents}
            if not actions:
                break
            obs, rewards, terminations, truncations, infos = env.step(actions)
            total_reward += sum(rewards.values())

            if not env.agents:
                break

        # Episode should complete without error; total_reward is finite
        assert np.isfinite(total_reward)
