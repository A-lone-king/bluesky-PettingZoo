"""Tests for BlueSkyMARLEnv action dispatch in SINGLE_RL mode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.types import ConflictConfig, SpawnConfig
from tests.helpers.env_factory import make_env


class SingleRLScenario(BaseScenario):
    """Minimal SINGLE_RL scenario for testing action dispatch."""

    def __init__(self):
        self._agents = [f"AC{i:03d}" for i in range(5)]

    @property
    def control_mode(self):
        return "SINGLE_RL"

    @property
    def ego_agent(self):
        return "AC000"

    @property
    def background_agents(self):
        return self._agents[1:]

    def setup(self, rng, airspace_bounds):
        return list(self._agents)

    def get_spawn_config(self):
        return SpawnConfig(altitude_range=(30000, 40000), speed_range=(400, 500), heading_range=(0, 360))

    def get_conflict_config(self):
        return ConflictConfig(nmac_horizontal_nm=5, nmac_vertical_ft=1000, warning_horizontal_nm=10, warning_vertical_ft=2000)

    def get_waypoint(self, agent_id):
        return {"lat": 40.0, "lon": 117.0, "alt": 35000.0, "hdg": 90.0}

    def get_background_actions(self, states):
        """Return default no-op actions for background agents."""
        return {aid: [2, 2, 2] for aid in self._background_agents if aid in states}

    @property
    def _background_agents(self):
        return self._agents[1:]


class TestEnvActionDispatch:
    """Verify SINGLE_RL action dispatch in step()."""

    def test_step_with_ego_action_only(self, tmp_path):
        """In SINGLE_RL mode, step() should only need ego agent's action."""
        scenario = SingleRLScenario()
        env = make_env(tmp_path=tmp_path, scenario=scenario, initial_count=5)
        env.reset(seed=42)

        # Only provide action for ego agent
        actions = {"AC000": [2, 2, 2]}
        obs, rewards, terms, truncs, infos = env.step(actions)
        # Should not crash
        assert "AC000" in obs

    def test_step_accepts_full_actions(self, tmp_path):
        """In SINGLE_RL mode, step() also works with all agents' actions."""
        scenario = SingleRLScenario()
        env = make_env(tmp_path=tmp_path, scenario=scenario, initial_count=5)
        env.reset(seed=42)

        actions = {aid: [2, 2, 2] for aid in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        assert "AC000" in obs

    def test_background_agents_get_default_actions(self, tmp_path):
        """Background agents should get default no-op actions when not provided."""
        scenario = SingleRLScenario()
        env = make_env(tmp_path=tmp_path, scenario=scenario, initial_count=5)
        env.reset(seed=42)

        # Only provide ego action
        actions = {"AC000": [2, 2, 2]}
        obs, rewards, terms, truncs, infos = env.step(actions)
        # Background agents should still have observations
        for aid in env.agents:
            if aid != "AC000":
                assert aid in obs
