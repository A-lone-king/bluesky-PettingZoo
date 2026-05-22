"""Tests for DescentScenario (T-D01).

Descent approach: multiple aircraft at cruising altitude must descend to
target altitudes before reaching the runway.  Vertical-speed-only control.
Crash (alt <= 0) truncates; runway arrival terminates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.descent import DescentScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from bluesky_pettingzoo.utils.types import AircraftState

from tests.helpers.fake_wrapper import FakeBlueSkyWrapper
from tests.helpers.env_factory import make_config as _make_config
from tests.helpers.env_factory import write_rewards_yaml as _write_rewards_yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env_with_scenario(
    env_config: dict[str, Any],
    scenario: DescentScenario,
) -> BlueSkyMARLEnv:
    """Create a BlueSkyMARLEnv with a DescentScenario."""
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
# T-D01 tests
# ===========================================================================


class TestDescentSetup:
    """Scenario initialization should return correct agent count."""

    def test_descent_setup_returns_agents(self, tmp_path: Path) -> None:
        """setup() returns the configured number of agent IDs."""
        scenario = DescentScenario(num_aircraft=3, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        agents = scenario.setup(rng, bounds)

        assert len(agents) == 3
        assert all(isinstance(a, str) for a in agents)
        assert len(set(agents)) == 3

    def test_descent_setup_unique_ids(self, tmp_path: Path) -> None:
        """All agent IDs must be unique."""
        scenario = DescentScenario(num_aircraft=5, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        agents = scenario.setup(rng, bounds)

        assert len(agents) == len(set(agents))


class TestDescentActionDimensions:
    """DescentScenario should restrict action space to vertical speed only."""

    def test_action_dimensions_vs_only(self) -> None:
        """action_dimensions returns [1] (altitude/vertical speed)."""
        scenario = DescentScenario(num_aircraft=3, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        scenario.setup(rng, bounds)

        assert scenario.action_dimensions == [1]


class TestDescentWaypoint:
    """Each agent should have a waypoint at the runway with a target altitude."""

    def test_waypoint_has_required_keys(self, tmp_path: Path) -> None:
        """get_waypoint() returns dict with lat, lon, alt, hdg."""
        scenario = DescentScenario(num_aircraft=2, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        agents = scenario.setup(rng, bounds)

        for agent_id in agents:
            wp = scenario.get_waypoint(agent_id)
            assert "lat" in wp, f"waypoint for {agent_id} missing 'lat'"
            assert "lon" in wp, f"waypoint for {agent_id} missing 'lon'"
            assert "alt" in wp, f"waypoint for {agent_id} missing 'alt'"
            assert "hdg" in wp, f"waypoint for {agent_id} missing 'hdg'"

    def test_waypoints_differ_between_agents(self, tmp_path: Path) -> None:
        """Different agents should have different target altitudes."""
        scenario = DescentScenario(num_aircraft=3, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        agents = scenario.setup(rng, bounds)

        alts = [scenario.get_waypoint(a)["alt"] for a in agents]
        # At least some altitudes should differ (not all identical)
        assert len(set(alts)) > 1, "Target altitudes should vary between agents"


class TestDescentCrashTruncation:
    """Aircraft at altitude <= 0 should be truncated (crash)."""

    def test_should_truncate_at_ground(self, tmp_path: Path) -> None:
        """Aircraft at alt=0 should be truncated."""
        scenario = DescentScenario(num_aircraft=2, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        agents = scenario.setup(rng, bounds)

        state = AircraftState(
            id=agents[0], lat=39.25, lon=116.25,
            alt=0.0, hdg=90.0, tas=150.0, vs=-5.0,
        )
        assert scenario.should_truncate(agents[0], state, bounds) is True

    def test_should_truncate_below_ground(self, tmp_path: Path) -> None:
        """Aircraft at alt < 0 should be truncated."""
        scenario = DescentScenario(num_aircraft=2, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        agents = scenario.setup(rng, bounds)

        state = AircraftState(
            id=agents[0], lat=39.25, lon=116.25,
            alt=-500.0, hdg=90.0, tas=150.0, vs=-10.0,
        )
        assert scenario.should_truncate(agents[0], state, bounds) is True

    def test_should_not_truncate_at_altitude(self, tmp_path: Path) -> None:
        """Aircraft at normal altitude should not be truncated."""
        scenario = DescentScenario(num_aircraft=2, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        agents = scenario.setup(rng, bounds)

        state = AircraftState(
            id=agents[0], lat=39.25, lon=116.25,
            alt=10000.0, hdg=90.0, tas=150.0, vs=-3.0,
        )
        assert scenario.should_truncate(agents[0], state, bounds) is False


class TestDescentSpawnConfig:
    """Spawn config should define descent-appropriate parameters."""

    def test_spawn_config_has_ranges(self) -> None:
        """get_spawn_config() returns valid SpawnConfig."""
        from bluesky_pettingzoo.utils.types import SpawnConfig

        scenario = DescentScenario(num_aircraft=2, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        scenario.setup(rng, bounds)

        spawn = scenario.get_spawn_config()
        assert isinstance(spawn, SpawnConfig)
        assert spawn.altitude_range[0] < spawn.altitude_range[1]
        assert spawn.speed_range[0] < spawn.speed_range[1]
        assert spawn.heading_range[0] < spawn.heading_range[1]


class TestDescentConflictConfig:
    """Conflict config should return standard thresholds."""

    def test_conflict_config_valid(self) -> None:
        """get_conflict_config() returns valid ConflictConfig."""
        from bluesky_pettingzoo.utils.types import ConflictConfig

        scenario = DescentScenario(num_aircraft=2, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        scenario.setup(rng, bounds)

        cfg = scenario.get_conflict_config()
        assert isinstance(cfg, ConflictConfig)
        assert cfg.nmac_horizontal_nm > 0
        assert cfg.nmac_vertical_ft > 0


class TestDescentFullEpisode:
    """Full episode with DescentScenario should run without errors."""

    def test_descent_full_episode(self, tmp_path: Path) -> None:
        """Run a complete episode with the scenario."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=3, max_steps=20)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = DescentScenario(num_aircraft=3, seed=42)
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

        assert np.isfinite(total_reward)


class TestDescentResetConsistency:
    """Reset should produce consistent initial state."""

    def test_reset_produces_observations(self, tmp_path: Path) -> None:
        """reset() returns observations for all agents."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=2)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = DescentScenario(num_aircraft=2, seed=42)
        env = _make_env_with_scenario(config, scenario)
        obs, infos = env.reset(seed=42)

        assert len(obs) == 2
        for agent_id in env.agents:
            assert agent_id in obs
            assert agent_id in infos
