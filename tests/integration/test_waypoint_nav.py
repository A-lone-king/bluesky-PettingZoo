"""Tests for WaypointNavScenario (T-V15).

Waypoint navigation: aircraft navigate to assigned waypoints without conflicts.
Used as a baseline for testing guidance logic and arrival termination.
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
from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario
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
    scenario: WaypointNavScenario,
) -> BlueSkyMARLEnv:
    """Create a BlueSkyMARLEnv with a WaypointNavScenario."""
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
# T-V15 tests
# ===========================================================================


class TestWaypointNavSetup:
    """Scenario initialization should succeed."""

    def test_waypoint_nav_setup(self) -> None:
        """setup() returns the configured number of agent IDs."""
        scenario = WaypointNavScenario(num_aircraft=3, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        agents = scenario.setup(rng, bounds)

        assert len(agents) == 3
        assert all(isinstance(a, str) for a in agents)
        assert len(set(agents)) == 3


class TestWaypointNavNoConflict:
    """Waypoint nav scenario should have no conflicts (aircraft far apart)."""

    def test_waypoint_nav_no_conflict(self, tmp_path: Path) -> None:
        """Aircraft are spaced far enough apart that no conflict is detected."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=3)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = WaypointNavScenario(num_aircraft=3, seed=42)
        env = _make_env_with_scenario(config, scenario)
        obs, infos = env.reset(seed=42)

        agents = list(env.agents)
        wrapper = env._wrapper

        # Verify aircraft are far apart (> 10 NM)
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                s1 = wrapper.get_aircraft_state(agents[i])
                s2 = wrapper.get_aircraft_state(agents[j])
                dist = haversine_distance(s1["lat"], s1["lon"], s2["lat"], s2["lon"])
                assert dist > 10.0, f"Aircraft too close: {dist:.1f} NM"


class TestWaypointNavArrival:
    """Aircraft reaching its waypoint should trigger termination."""

    def test_waypoint_nav_arrival(self, tmp_path: Path) -> None:
        """Aircraft at its waypoint terminates."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=2)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = WaypointNavScenario(num_aircraft=2, seed=42)
        env = _make_env_with_scenario(config, scenario)
        env.reset(seed=42)

        agents = list(env.agents)
        wrapper = env._wrapper

        # Place the first aircraft at its waypoint
        wp = scenario.get_waypoint(agents[0])
        wrapper.set_aircraft_state(agents[0], lat=wp["lat"], lon=wp["lon"], alt=wp["alt"])

        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, terminations, _, _ = env.step(actions)

        assert terminations[agents[0]] is True


class TestWaypointNavGuidance:
    """Aircraft should fly toward their assigned waypoints."""

    def test_waypoint_nav_guidance(self, tmp_path: Path) -> None:
        """After several steps, aircraft moves closer to its waypoint."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=2, max_steps=100)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = WaypointNavScenario(num_aircraft=2, seed=42)
        env = _make_env_with_scenario(config, scenario)
        env.reset(seed=42)

        agents = list(env.agents)
        wrapper = env._wrapper
        acid = agents[0]
        wp = scenario.get_waypoint(acid)

        # Set aircraft heading to point toward its waypoint
        wrapper.set_aircraft_state(acid, hdg=wp["hdg"])

        # Get initial distance to waypoint
        st = wrapper.get_aircraft_state(acid)
        initial_dist = haversine_distance(st["lat"], st["lon"], wp["lat"], wp["lon"])

        # Run several steps with no-op actions
        for _ in range(10):
            actions = {a: [2, 2, 2] for a in env.agents}
            if not actions:
                break
            env.step(actions)
            if acid not in env.agents:
                break

        if acid in env.agents:
            st = wrapper.get_aircraft_state(acid)
            final_dist = haversine_distance(st["lat"], st["lon"], wp["lat"], wp["lon"])
            # Aircraft heading toward waypoint should get closer
            assert final_dist < initial_dist


class TestWaypointNavFullEpisode:
    """Full episode with WaypointNavScenario should run without errors."""

    def test_waypoint_nav_full_episode(self, tmp_path: Path) -> None:
        """Run a complete episode with the scenario."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=3, max_steps=30)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = WaypointNavScenario(num_aircraft=3, seed=42)
        env = _make_env_with_scenario(config, scenario)
        obs, infos = env.reset(seed=42)

        assert len(env.agents) == 3

        total_reward = 0.0
        for step in range(30):
            actions = {a: [2, 2, 2] for a in env.agents}
            if not actions:
                break
            obs, rewards, terminations, truncations, infos = env.step(actions)
            total_reward += sum(rewards.values())

            if not env.agents:
                break

        assert np.isfinite(total_reward)
