"""Tests for DescentScenario SINGLE_RL control mode."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_pettingzoo.envs.scenarios.descent import DescentScenario


@pytest.fixture
def scenario():
    s = DescentScenario(num_aircraft=5, seed=42)
    bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
    rng = np.random.RandomState(42)
    s.setup(rng, bounds)
    return s


class TestDescentControlMode:
    """Verify DescentScenario SINGLE_RL mode."""

    def test_control_mode_is_single_rl(self, scenario):
        assert scenario.control_mode == "SINGLE_RL"

    def test_ego_agent_is_ac000(self, scenario):
        assert scenario.ego_agent == "AC000"

    def test_background_agents_count(self, scenario):
        bg = scenario.background_agents
        assert len(bg) == 4  # 5 total - 1 ego

    def test_ego_not_in_background(self, scenario):
        bg = scenario.background_agents
        assert "AC000" not in bg

    def test_background_agents_are_remaining(self, scenario):
        bg = scenario.background_agents
        assert bg == [f"AC{i:03d}" for i in range(1, 5)]
