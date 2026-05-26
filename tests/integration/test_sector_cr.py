"""Tests for SectorCRScenario (T-V11).

Sector conflict resolution: multiple aircraft inside a polygon sector,
use heading + speed maneuvers to avoid conflicts.
Aircraft leaving the sector are truncated (not terminated).
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
from bluesky_pettingzoo.envs.scenarios.sector_cr import SectorCRScenario
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
    scenario: SectorCRScenario,
) -> BlueSkyMARLEnv:
    """Create a BlueSkyMARLEnv with a SectorCRScenario."""
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
# T-V11 tests
# ===========================================================================


class TestSectorCRSetup:
    """Scenario initialization should succeed."""

    def test_sector_cr_setup(self) -> None:
        """setup() returns the configured number of agent IDs."""
        scenario = SectorCRScenario(num_aircraft=5, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        agents = scenario.setup(rng, bounds)

        assert len(agents) == 5
        assert all(isinstance(a, str) for a in agents)
        assert len(set(agents)) == 5


class TestSectorCRPolygonBoundary:
    """Sector boundary should be a polygon (not just a rectangle)."""

    def test_sector_cr_polygon_boundary(self) -> None:
        """Scenario defines a polygon with at least 3 vertices."""
        scenario = SectorCRScenario(num_aircraft=5, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        scenario.setup(rng, bounds)

        polygon = scenario.get_sector_polygon()
        assert len(polygon) >= 3, "Polygon must have at least 3 vertices"
        # Each vertex should be (lat, lon) tuple
        for vertex in polygon:
            assert len(vertex) == 2


class TestSectorCRExitTruncation:
    """Aircraft leaving the sector should be truncated (not terminated)."""

    def test_sector_cr_exit_truncation(self, tmp_path: Path) -> None:
        """Aircraft outside polygon is truncated by scenario."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=3)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = SectorCRScenario(num_aircraft=3, seed=42)
        env = _make_env_with_scenario(config, scenario)
        env.reset(seed=42)

        agents = list(env.agents)
        wrapper = env._wrapper

        # Place one aircraft well outside the polygon
        # The polygon is centered around mid_lat/mid_lon, so placing
        # an aircraft at the edge of the bounding box should be outside
        wrapper.set_aircraft_state(agents[0], lat=39.0, lon=116.0, alt=35000, hdg=90.0)

        initial_agents = set(env.agents)
        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, terminations, truncations, _ = env.step(actions)

        # The out-of-sector aircraft should be truncated
        assert truncations[agents[0]] is True


class TestSectorCRConflictDetection:
    """Conflict detection should work inside the sector."""

    def test_sector_cr_conflict_detection(self, tmp_path: Path) -> None:
        """Two aircraft close together are detected as conflicting."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=2)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = SectorCRScenario(num_aircraft=2, seed=42)
        env = _make_env_with_scenario(config, scenario)
        env.reset(seed=42)

        agents = list(env.agents)
        wrapper = env._wrapper

        # Place aircraft close together (within NMAC distance)
        mid_lat = 39.25
        mid_lon = 116.25
        wrapper.set_aircraft_state(agents[0], lat=mid_lat, lon=mid_lon, alt=35000, hdg=90.0)
        wrapper.set_aircraft_state(agents[1], lat=mid_lat, lon=mid_lon + 0.03, alt=35000, hdg=270.0)

        dist = haversine_distance(mid_lat, mid_lon, mid_lat, mid_lon + 0.03)
        assert dist < 5.0  # Within NMAC distance

        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, _, _, infos = env.step(actions)

        for agent_id in agents:
            if agent_id in infos:
                ts = infos[agent_id].get("textual_state", {})
                assert ts.get("conflict_status") in ("warning", "nmac")


class TestSectorCRActionHdgSpd:
    """SectorCR scenario should allow heading + speed actions."""

    def test_sector_cr_action_hdg_spd(self) -> None:
        """Scenario reports heading + speed as valid action dimensions."""
        scenario = SectorCRScenario(num_aircraft=3, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        scenario.setup(rng, bounds)

        # action_dimensions = [0, 2] = heading + speed
        assert scenario.action_dimensions == [0, 2]


class TestSectorCRDensityConfigurable:
    """Aircraft density should be configurable."""

    def test_sector_cr_density_configurable(self) -> None:
        """Different density settings produce different aircraft counts."""
        scenario_low = SectorCRScenario(num_aircraft=3, seed=42)
        scenario_high = SectorCRScenario(num_aircraft=10, seed=42)

        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}

        agents_low = scenario_low.setup(np.random.RandomState(42), bounds)
        agents_high = scenario_high.setup(np.random.RandomState(42), bounds)

        assert len(agents_low) < len(agents_high)


class TestSectorCRFullEpisode:
    """Full episode with SectorCRScenario should run without errors."""

    def test_sector_cr_full_episode(self, tmp_path: Path) -> None:
        """Run a complete episode with the scenario."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=5, max_steps=20)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = SectorCRScenario(num_aircraft=5, seed=42)
        env = _make_env_with_scenario(config, scenario)
        obs, infos = env.reset(seed=42)

        assert len(env.agents) == 5

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


# ===========================================================================
# Bug fix: SectorCR aircraft must spawn INSIDE the polygon
# ===========================================================================


class TestSectorCRInitialPositions:
    """Scenario should provide initial positions inside the polygon."""

    def test_get_initial_positions_returns_dict(self) -> None:
        """After setup(), get_initial_positions() returns a dict mapping agent IDs to (lat, lon)."""
        from bluesky_pettingzoo.utils.geometry import point_in_polygon

        scenario = SectorCRScenario(num_aircraft=3, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        agents = scenario.setup(rng, bounds)

        positions = scenario.get_initial_positions()
        assert positions is not None
        assert isinstance(positions, dict)
        assert set(positions.keys()) == set(agents)

        polygon = scenario.get_sector_polygon()
        for acid, (lat, lon, _alt) in positions.items():
            assert point_in_polygon(lat, lon, polygon), (
                f"{acid} position ({lat}, {lon}) is outside the polygon"
            )

    def test_get_initial_positions_returns_none_before_setup(self) -> None:
        """get_initial_positions() returns None if setup() hasn't been called."""
        scenario = SectorCRScenario(num_aircraft=3, seed=42)
        assert scenario.get_initial_positions() is None


class TestSectorCREpisodeLength:
    """SectorCR episodes must last more than 1 step (bug regression)."""

    def test_episode_runs_multiple_steps(self, tmp_path: Path) -> None:
        """Aircraft inside the polygon should not be truncated on step 1."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=3, max_steps=20)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = SectorCRScenario(num_aircraft=3, seed=42)
        env = _make_env_with_scenario(config, scenario)
        env.reset(seed=42)

        assert len(env.agents) == 3

        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, _, truncations, _ = env.step(actions)

        # At least some agents should survive step 1 (not all truncated)
        surviving = [a for a in env.agents]
        assert len(surviving) > 0, (
            "All agents truncated on step 1 -- aircraft likely spawned outside polygon"
        )

    def test_episode_lasting_multiple_steps(self, tmp_path: Path) -> None:
        """Episode should last more than 1 step overall."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=3, max_steps=20)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = SectorCRScenario(num_aircraft=3, seed=42)
        env = _make_env_with_scenario(config, scenario)
        env.reset(seed=42)

        step_count = 0
        for _ in range(20):
            if not env.agents:
                break
            actions = {a: [2, 2, 2] for a in env.agents}
            env.step(actions)
            step_count += 1

        assert step_count > 1, (
            f"Episode lasted only {step_count} step(s) -- expected > 1"
        )
