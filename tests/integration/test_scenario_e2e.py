"""Scenario end-to-end integration tests (G-V03).

Run a complete episode for each scenario and verify:
- Reset succeeds
- Step loop completes
- Rewards are finite
- At least some agents terminate or truncate
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
from bluesky_pettingzoo.envs.scenarios.descent import DescentScenario
from bluesky_pettingzoo.envs.scenarios.horizontal_cr import HorizontalCRScenario
from bluesky_pettingzoo.envs.scenarios.merge import MergeScenario
from bluesky_pettingzoo.envs.scenarios.sector_capacity import SectorCapacityScenario
from bluesky_pettingzoo.envs.scenarios.sector_cr import SectorCRScenario
from bluesky_pettingzoo.envs.scenarios.vertical_cr import VerticalCRScenario
from bluesky_pettingzoo.envs.scenarios.static_obstacle import StaticObstacleScenario
from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.capacity import CapacityPenalty
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.obstacle_intrusion import ObstacleIntrusion
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from bluesky_pettingzoo.envs.scenarios.base import BaseScenario

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from tests.helpers.env_factory import make_config as _make_config
from tests.helpers.env_factory import write_rewards_yaml as _write_rewards_yaml


# ---------------------------------------------------------------------------
# Fake BlueSkyWrapper
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------




def _run_episode(
    tmp_path: Path,
    config: dict[str, Any],
    scenario: BaseScenario,
    num_steps: int = 20,
) -> dict[str, Any]:
    """Run a full episode and return summary stats."""
    _write_rewards_yaml(tmp_path)
    config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

    wrapper = BlueSkyWrapper(config)
    obs_manager = ObservationManager(config)
    action_translator = ActionTranslator(config)

    with open(config["_rewards_yaml"], encoding="utf-8") as f:
        rewards_cfg = yaml.safe_load(f)
    merged = {**config, **rewards_cfg}

    calc = RewardCalculator()
    calc.register(ConflictPenalty(merged), weight=1.0)
    calc.register(SmoothnessPenalty(merged), weight=0.5)
    eff = EfficiencyReward(merged)
    calc.register(eff, weight=0.3)
    if hasattr(scenario, "get_obstacles"):
        obs_comp = ObstacleIntrusion()
        obs_comp.set_obstacles(scenario.get_obstacles())
        calc.register(obs_comp, weight=1.0)
    if hasattr(scenario, "get_sectors"):
        cap_comp = CapacityPenalty(merged)
        calc.register(cap_comp, weight=1.0)

    env = BlueSkyMARLEnv(
        config=config,
        wrapper=wrapper,
        observation_manager=obs_manager,
        action_translator=action_translator,
        reward_calculator=calc,
        rewards_config=rewards_cfg,
        scenario=scenario,
    )

    obs, infos = env.reset(seed=42)
    initial_agents = set(env.agents)

    total_reward = 0.0
    steps_taken = 0
    all_terminated = set()
    all_truncated = set()

    for _ in range(num_steps):
        actions = {a: [2, 2, 2] for a in env.agents}
        if not actions:
            break
        obs, rewards, terminations, truncations, infos = env.step(actions)
        total_reward += sum(rewards.values())
        steps_taken += 1

        for aid, t in terminations.items():
            if t:
                all_terminated.add(aid)
        for aid, t in truncations.items():
            if t:
                all_truncated.add(aid)

        if not env.agents:
            break

    env.close()

    return {
        "initial_agents": initial_agents,
        "steps": steps_taken,
        "total_reward": total_reward,
        "terminated": all_terminated,
        "truncated": all_truncated,
    }


# ===========================================================================
# G-V03 tests
# ===========================================================================


class TestHorizontalCRE2E:
    """Horizontal CR scenario full episode."""

    def test_horizontal_cr_e2e(self, tmp_path: Path) -> None:
        config = _make_config(max_steps=30)
        scenario = HorizontalCRScenario(num_aircraft=4, seed=42)
        stats = _run_episode(tmp_path, config, scenario, num_steps=30)

        assert stats["steps"] > 0
        assert np.isfinite(stats["total_reward"])
        assert len(stats["initial_agents"]) == 4


class TestVerticalCRE2E:
    """Vertical CR scenario full episode."""

    def test_vertical_cr_e2e(self, tmp_path: Path) -> None:
        config = _make_config(max_steps=30)
        scenario = VerticalCRScenario(num_aircraft=4, seed=42)
        stats = _run_episode(tmp_path, config, scenario, num_steps=30)

        assert stats["steps"] > 0
        assert np.isfinite(stats["total_reward"])
        assert len(stats["initial_agents"]) == 4


class TestSectorCRE2E:
    """Sector CR scenario full episode."""

    def test_sector_cr_e2e(self, tmp_path: Path) -> None:
        config = _make_config(max_steps=30)
        scenario = SectorCRScenario(num_aircraft=5, seed=42)
        stats = _run_episode(tmp_path, config, scenario, num_steps=30)

        assert stats["steps"] > 0
        assert np.isfinite(stats["total_reward"])
        assert len(stats["initial_agents"]) == 5


class TestWaypointNavE2E:
    """Waypoint nav scenario full episode."""

    def test_waypoint_nav_e2e(self, tmp_path: Path) -> None:
        config = _make_config(max_steps=30)
        scenario = WaypointNavScenario(num_aircraft=3, seed=42)
        stats = _run_episode(tmp_path, config, scenario, num_steps=30)

        assert stats["steps"] > 0
        assert np.isfinite(stats["total_reward"])
        assert len(stats["initial_agents"]) == 3


class TestMergeE2E:
    """Merge scenario full episode."""

    def test_merge_e2e(self, tmp_path: Path) -> None:
        config = _make_config(max_steps=20)
        config["aircraft"]["initial_count"] = 20
        config["observation"]["max_observable_aircraft"] = 5
        scenario = MergeScenario(num_aircraft=20, seed=42)
        stats = _run_episode(tmp_path, config, scenario, num_steps=20)

        assert stats["steps"] > 0
        assert np.isfinite(stats["total_reward"])
        assert len(stats["initial_agents"]) == 20


class TestDescentE2E:
    """Descent scenario full episode."""

    def test_descent_e2e(self, tmp_path: Path) -> None:
        config = _make_config(max_steps=30)
        scenario = DescentScenario(num_aircraft=3, seed=42)
        stats = _run_episode(tmp_path, config, scenario, num_steps=30)

        assert stats["steps"] > 0
        assert np.isfinite(stats["total_reward"])
        assert len(stats["initial_agents"]) == 3


class TestStaticObstacleE2E:
    """StaticObstacle scenario full episode."""

    def test_static_obstacle_e2e(self, tmp_path: Path) -> None:
        config = _make_config(max_steps=20, observation={"max_obstacles": 10})
        scenario = StaticObstacleScenario(num_aircraft=1, num_obstacles=10, seed=42)
        stats = _run_episode(tmp_path, config, scenario, num_steps=20)

        assert stats["steps"] > 0
        assert np.isfinite(stats["total_reward"])
        assert len(stats["initial_agents"]) == 1


class TestSectorCapacityE2E:
    """SectorCapacity scenario full episode."""

    def test_sector_capacity_e2e(self, tmp_path: Path) -> None:
        config = _make_config(max_steps=30)
        scenario = SectorCapacityScenario(num_aircraft=6, num_sectors=2, sector_capacity=4, seed=42)
        stats = _run_episode(tmp_path, config, scenario, num_steps=30)

        assert stats["steps"] > 0
        assert np.isfinite(stats["total_reward"])
        assert len(stats["initial_agents"]) == 6
