"""Tests for RandomAgent."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from bluesky_pettingzoo.agents.random_agent import RandomAgent


@pytest.fixture
def action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([5, 5, 5])


@pytest.fixture
def agent() -> RandomAgent:
    return RandomAgent()


class TestActReturnsDict:
    """act() must return a dict."""

    def test_act_returns_dict(
        self,
        agent: RandomAgent,
        action_space: spaces.MultiDiscrete,
    ) -> None:
        obs = {"AC001": np.zeros(6, dtype=np.float32)}
        spaces_map = {"AC001": action_space}

        result = agent.act(obs, spaces_map)

        assert isinstance(result, dict)


class TestActKeysMatchAgents:
    """Returned dict keys must match input agent IDs."""

    def test_act_keys_match_agents(
        self,
        agent: RandomAgent,
        action_space: spaces.MultiDiscrete,
    ) -> None:
        agents = ["A", "B", "C"]
        obs = {a: np.zeros(6, dtype=np.float32) for a in agents}
        spaces_map = {a: action_space for a in agents}

        result = agent.act(obs, spaces_map)

        assert set(result.keys()) == set(agents)


class TestActionInSpace:
    """Each action must be valid within its action space."""

    def test_action_in_space(
        self,
        agent: RandomAgent,
        action_space: spaces.MultiDiscrete,
    ) -> None:
        obs = {"AC001": np.zeros(6, dtype=np.float32)}
        spaces_map = {"AC001": action_space}

        # Sample 100 times; every action must be valid
        for _ in range(100):
            result = agent.act(obs, spaces_map)
            action = result["AC001"]
            assert action_space.contains(action)


class TestDifferentObservationsDifferentActions:
    """Different random seeds / calls should produce varied actions."""

    def test_different_observations_different_actions(
        self,
        agent: RandomAgent,
        action_space: spaces.MultiDiscrete,
    ) -> None:
        obs = {"AC001": np.zeros(6, dtype=np.float32)}
        spaces_map = {"AC001": action_space}

        actions = set()
        for _ in range(50):
            result = agent.act(obs, spaces_map)
            actions.add(tuple(result["AC001"]))

        # With 125 possible outcomes, 50 samples should yield >1 unique value
        assert len(actions) > 1


class TestReset:
    """reset() must not raise."""

    def test_reset_no_error(self, agent: RandomAgent) -> None:
        agent.reset()
