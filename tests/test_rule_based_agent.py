"""Tests for RuleBasedAgent (straight-flight, no avoidance)."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from bluesky_pettingzoo.agents.rule_based_agent import RuleBasedAgent


@pytest.fixture
def action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([5, 5, 5])


@pytest.fixture
def agent() -> RuleBasedAgent:
    return RuleBasedAgent()


class TestActReturnsDict:
    """act() must return a dict."""

    def test_act_returns_dict(
        self, agent: RuleBasedAgent, action_space: spaces.MultiDiscrete,
    ) -> None:
        obs = {"AC001": np.zeros(6, dtype=np.float32)}
        spaces_map = {"AC001": action_space}

        result = agent.act(obs, spaces_map)

        assert isinstance(result, dict)


class TestActKeysMatchAgents:
    """Returned dict keys must match input agent IDs."""

    def test_act_keys_match_agents(
        self, agent: RuleBasedAgent, action_space: spaces.MultiDiscrete,
    ) -> None:
        agents = ["A", "B", "C"]
        obs = {a: np.zeros(6, dtype=np.float32) for a in agents}
        spaces_map = {a: action_space for a in agents}

        result = agent.act(obs, spaces_map)

        assert set(result.keys()) == set(agents)


class TestAlwaysNoAdjustment:
    """Action must always be [2, 2, 2] (zero adjustment)."""

    def test_always_no_adjustment(
        self, agent: RuleBasedAgent, action_space: spaces.MultiDiscrete,
    ) -> None:
        obs = {"AC001": np.zeros(6, dtype=np.float32)}
        spaces_map = {"AC001": action_space}

        result = agent.act(obs, spaces_map)

        assert list(result["AC001"]) == [2, 2, 2]


class TestDeterministic:
    """Same input must always produce the same output."""

    def test_deterministic(
        self, agent: RuleBasedAgent, action_space: spaces.MultiDiscrete,
    ) -> None:
        agents = ["X", "Y"]
        obs = {a: np.random.randn(6).astype(np.float32) for a in agents}
        spaces_map = {a: action_space for a in agents}

        r1 = agent.act(obs, spaces_map)
        r2 = agent.act(obs, spaces_map)

        for aid in agents:
            assert list(r1[aid]) == list(r2[aid])


class TestReset:
    """reset() must not raise."""

    def test_reset_no_error(self, agent: RuleBasedAgent) -> None:
        agent.reset()
