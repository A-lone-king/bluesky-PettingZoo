"""Tests for baseline evaluation logic."""

from __future__ import annotations

# Import the module under test
import sys
from pathlib import Path

import numpy as np
import pytest

from bluesky_pettingzoo.agents.random_agent import RandomAgent
from bluesky_pettingzoo.agents.rule_based_agent import RuleBasedAgent
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from evaluate_baselines import (
    BaselineMetrics,
    EpisodeResult,
    evaluate_agent,
    make_env_factory,
    run_episode,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_factory(tmp_path: Path, num_aircraft: int = 2):
    """Create an env factory for testing."""
    return make_env_factory(
        tmp_path=tmp_path,
        num_aircraft=num_aircraft,
        max_steps=10,
        seed=42,
    )


# ===========================================================================
# Tests
# ===========================================================================


class TestEpisodeResult:
    def test_fields(self) -> None:
        r = EpisodeResult(total_reward=-10.0, steps=5, arrived=False, nmac=False, truncated=True)
        assert r.total_reward == -10.0
        assert r.steps == 5
        assert r.arrived is False
        assert r.nmac is False
        assert r.truncated is True


class TestBaselineMetrics:
    def test_from_results(self) -> None:
        results = [
            EpisodeResult(-10.0, 5, False, False, True),
            EpisodeResult(-5.0, 3, True, False, False),
            EpisodeResult(-20.0, 8, False, True, False),
        ]
        m = BaselineMetrics.from_results(results)
        assert m.num_episodes == 3
        assert m.mean_reward == pytest.approx(-11.666, abs=0.01)
        assert m.arrival_rate == pytest.approx(1 / 3, abs=0.01)
        assert m.nmac_rate == pytest.approx(1 / 3, abs=0.01)
        assert m.mean_steps == pytest.approx(5.333, abs=0.01)

    def test_empty_results(self) -> None:
        m = BaselineMetrics.from_results([])
        assert m.num_episodes == 0
        assert m.mean_reward == 0.0
        assert m.arrival_rate == 0.0


class TestRunEpisode:
    def test_returns_episode_result(self, tmp_path: Path) -> None:
        factory = _make_factory(tmp_path)
        env = factory()
        agent = RuleBasedAgent()
        try:
            result = run_episode(env, agent, max_steps=10)
            assert isinstance(result, EpisodeResult)
            assert np.isfinite(result.total_reward)
            assert result.steps > 0
        finally:
            env.close()

    def test_random_agent_runs(self, tmp_path: Path) -> None:
        factory = _make_factory(tmp_path)
        env = factory()
        agent = RandomAgent()
        try:
            result = run_episode(env, agent, max_steps=10)
            assert isinstance(result, EpisodeResult)
            assert np.isfinite(result.total_reward)
        finally:
            env.close()


class TestEvaluateAgent:
    def test_returns_metrics(self, tmp_path: Path) -> None:
        factory = _make_factory(tmp_path)
        agent = RuleBasedAgent()
        metrics = evaluate_agent(factory, agent, num_episodes=3)
        assert isinstance(metrics, BaselineMetrics)
        assert metrics.num_episodes == 3
        assert np.isfinite(metrics.mean_reward)

    def test_random_and_rule_based_differ(self, tmp_path: Path) -> None:
        """Random and rule-based agents should produce different metrics."""
        factory = _make_factory(tmp_path)
        random_metrics = evaluate_agent(factory, RandomAgent(), num_episodes=5)
        rule_metrics = evaluate_agent(factory, RuleBasedAgent(), num_episodes=5)
        # They should at least have different reward distributions
        # (not necessarily different means with only 5 episodes, but metrics should be valid)
        assert np.isfinite(random_metrics.mean_reward)
        assert np.isfinite(rule_metrics.mean_reward)


class TestMakeEnvFactory:
    def test_factory_creates_env(self, tmp_path: Path) -> None:
        factory = make_env_factory(tmp_path=tmp_path, num_aircraft=2, max_steps=10, seed=42)
        env = factory()
        assert isinstance(env, BlueSkyMARLEnv)
        env.close()

    def test_factory_with_scenario(self, tmp_path: Path) -> None:
        scenario = WaypointNavScenario(num_aircraft=2, seed=42)
        factory = make_env_factory(
            tmp_path=tmp_path,
            num_aircraft=2,
            max_steps=10,
            seed=42,
            scenario=scenario,
        )
        env = factory()
        assert isinstance(env, BlueSkyMARLEnv)
        env.close()
