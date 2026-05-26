"""Tests for ModelEvaluator — strategy evaluation and comparison (A5)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_mock_obs() -> dict:
    """Create a valid observation dict compatible with RuleBasedAgent."""
    return {
        "self_state": np.zeros(9, dtype=np.float32),
        "other_aircraft": np.zeros((10, 10), dtype=np.float32),
        "other_aircraft_mask": np.zeros(10, dtype=np.int8),
        "goal": np.zeros(4, dtype=np.float32),
    }


def _make_mock_env():
    """Create a mock gymnasium env that returns valid obs/reward/terminated/truncated/info."""
    env = MagicMock()
    single_obs = _make_mock_obs()
    multi_obs = {"AC000": single_obs}
    env.reset.return_value = (multi_obs, {})
    env.observation_space = MagicMock()
    env.action_space = MagicMock()
    env.action_space.sample.return_value = np.array([2, 2, 2])
    env.agents = ["AC000"]
    # step returns (obs, rewards, terminations, truncations, infos) for multi-agent
    env.step.return_value = (multi_obs, {"AC000": -1.0}, {"AC000": False}, {"AC000": False}, {"AC000": {}})
    return env


def _make_single_agent_mock_env():
    """Create a mock that simulates SingleAgentGymWrapper (no .agents attribute).

    SingleAgentGymWrapper wraps a ParallelEnv. It has ._env (with .agents)
    and ._ego, but no .agents directly. Its .step() takes a single action
    and returns (obs, reward, terminated, truncated, info).
    """
    env = MagicMock(spec=[])  # no auto-created attributes
    single_obs = _make_mock_obs()

    # SingleAgentGymWrapper-like interface
    env.observation_space = MagicMock()
    env.action_space = MagicMock()
    env.action_space.sample.return_value = np.array([2, 2, 2])
    env.reset = MagicMock(return_value=(single_obs, {}))
    env.step = MagicMock(return_value=(single_obs, -1.0, False, False, {}))
    env.close = MagicMock()

    # Inner env with .agents
    env._env = MagicMock()
    env._env.agents = ["AC000"]
    env._ego = "AC000"

    return env


def _make_env_factory():
    """Return a factory that creates mock envs."""
    def factory():
        return _make_mock_env()
    return factory


class TestModelEvaluator:
    """ModelEvaluator should evaluate strategies and produce valid results."""

    def test_evaluate_random_returns_valid_result(self, tmp_path: Path) -> None:
        """Random strategy evaluation should return a valid EvalResult."""
        from bluesky_pettingzoo.training.evaluator import ModelEvaluator

        evaluator = ModelEvaluator(
            env_factory=_make_env_factory(),
            num_episodes=3,
            max_steps=10,
            seed=42,
        )
        result = evaluator.evaluate_random()
        assert result.strategy == "Random"
        assert result.num_episodes == 3
        assert isinstance(result.mean_reward, float)
        assert isinstance(result.std_reward, float)

    def test_evaluate_rule_based_returns_valid_result(self, tmp_path: Path) -> None:
        """RuleBased strategy evaluation should return a valid EvalResult."""
        from bluesky_pettingzoo.training.evaluator import ModelEvaluator

        evaluator = ModelEvaluator(
            env_factory=_make_env_factory(),
            num_episodes=3,
            max_steps=10,
            seed=42,
        )
        result = evaluator.evaluate_rule_based()
        assert result.strategy == "RuleBased"
        assert result.num_episodes == 3

    def test_result_fields_in_range(self, tmp_path: Path) -> None:
        """EvalResult fields should be in reasonable ranges."""
        from bluesky_pettingzoo.training.evaluator import ModelEvaluator

        evaluator = ModelEvaluator(
            env_factory=_make_env_factory(),
            num_episodes=5,
            max_steps=10,
            seed=42,
        )
        result = evaluator.evaluate_random()
        assert 0.0 <= result.arrival_rate <= 1.0
        assert 0.0 <= result.nmac_rate <= 1.0
        assert result.mean_steps >= 0
        assert result.std_reward >= 0.0

    def test_compare_all_returns_three_results(self, tmp_path: Path) -> None:
        """compare_all should return 3 EvalResult objects."""
        from bluesky_pettingzoo.training.evaluator import ModelEvaluator

        evaluator = ModelEvaluator(
            env_factory=_make_env_factory(),
            num_episodes=3,
            max_steps=10,
            seed=42,
        )
        results = evaluator.compare_all()
        assert len(results) == 3
        strategies = {r.strategy for r in results}
        assert strategies == {"Random", "RuleBased", "PPO"}

    def test_format_table_output(self, tmp_path: Path) -> None:
        """format_table should include all strategy names."""
        from bluesky_pettingzoo.training.evaluator import ModelEvaluator, EvalResult

        results = [
            EvalResult(strategy="Random", scenario="Test", mean_reward=-10.0, std_reward=5.0,
                       min_reward=-20.0, max_reward=-2.0, mean_steps=30.0,
                       arrival_rate=0.0, nmac_rate=0.5, num_episodes=10, seed=42),
            EvalResult(strategy="RuleBased", scenario="Test", mean_reward=-5.0, std_reward=3.0,
                       min_reward=-15.0, max_reward=0.0, mean_steps=25.0,
                       arrival_rate=0.2, nmac_rate=0.1, num_episodes=10, seed=42),
            EvalResult(strategy="PPO", scenario="Test", mean_reward=-3.0, std_reward=2.0,
                       min_reward=-10.0, max_reward=1.0, mean_steps=20.0,
                       arrival_rate=0.5, nmac_rate=0.0, num_episodes=10, seed=42),
        ]
        table = ModelEvaluator.format_table(results)
        assert "Random" in table
        assert "RuleBased" in table
        assert "PPO" in table

    def test_evaluate_rule_based_with_single_agent_wrapper(self, tmp_path: Path) -> None:
        """RuleBased should work with SingleAgentGymWrapper-like env (no .agents)."""
        from bluesky_pettingzoo.training.evaluator import ModelEvaluator

        def factory():
            return _make_single_agent_mock_env()

        evaluator = ModelEvaluator(
            env_factory=factory,
            num_episodes=3,
            max_steps=10,
            seed=42,
        )
        result = evaluator.evaluate_rule_based()
        assert result.strategy == "RuleBased"
        assert result.num_episodes == 3
        # Must have actually run steps, not returned 0
        assert result.mean_steps > 0

    def test_evaluate_random_with_single_agent_wrapper(self, tmp_path: Path) -> None:
        """Random should work with SingleAgentGymWrapper-like env."""
        from bluesky_pettingzoo.training.evaluator import ModelEvaluator

        def factory():
            return _make_single_agent_mock_env()

        evaluator = ModelEvaluator(
            env_factory=factory,
            num_episodes=3,
            max_steps=10,
            seed=42,
        )
        result = evaluator.evaluate_random()
        assert result.strategy == "Random"
        assert result.mean_steps > 0

    def test_deterministic_with_seed(self, tmp_path: Path) -> None:
        """Same seed should produce same random evaluation results."""
        from bluesky_pettingzoo.training.evaluator import ModelEvaluator

        def make_evaluator():
            return ModelEvaluator(
                env_factory=_make_env_factory(),
                num_episodes=3,
                max_steps=10,
                seed=42,
            )

        r1 = make_evaluator().evaluate_random()
        r2 = make_evaluator().evaluate_random()
        assert r1.mean_reward == pytest.approx(r2.mean_reward, abs=1e-6)
