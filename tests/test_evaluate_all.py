"""Tests for evaluate_all script — multi-algorithm evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestEvaluateAllImport:
    """Verify evaluate_all module can be imported."""

    def test_import(self):
        from scripts.evaluate_all import evaluate_all_scenarios
        assert evaluate_all_scenarios is not None

    def test_has_main(self):
        from scripts.evaluate_all import main
        assert callable(main)


class TestEvaluateAllScenarios:
    """Verify evaluate_all_scenarios runs all algorithms."""

    @patch("scripts.evaluate_all.ModelEvaluator")
    def test_returns_results_for_all_algorithms(self, mock_evaluator_cls):
        """Should return results for Random, RuleBased, PPO, SAC, TD3, DDPG."""
        mock_evaluator = MagicMock()
        mock_evaluator_cls.return_value = mock_evaluator

        mock_result = MagicMock()
        mock_result.strategy = "PPO"
        mock_result.scenario = "HorizontalCR"
        mock_result.mean_reward = -10.0
        mock_result.std_reward = 5.0
        mock_result.min_reward = -20.0
        mock_result.max_reward = -5.0
        mock_result.mean_steps = 50.0
        mock_result.arrival_rate = 0.1
        mock_result.nmac_rate = 0.0
        mock_result.num_episodes = 10
        mock_result.seed = 42

        mock_evaluator.evaluate_random.return_value = mock_result
        mock_evaluator.evaluate_rule_based.return_value = mock_result
        mock_evaluator.evaluate_ppo.return_value = mock_result
        mock_evaluator.evaluate_sac.return_value = mock_result
        mock_evaluator.evaluate_td3.return_value = mock_result
        mock_evaluator.evaluate_ddpg.return_value = mock_result

        from scripts.evaluate_all import evaluate_all_scenarios
        results = evaluate_all_scenarios(
            scenario_name="HorizontalCR",
            num_aircraft=3,
            max_steps=50,
            num_episodes=5,
            model_dir=Path("models"),
        )

        # Should have results for all 6 strategies
        assert len(results) == 6

    @patch("scripts.evaluate_all.ModelEvaluator")
    def test_handles_missing_models(self, mock_evaluator_cls):
        """Should handle missing model files gracefully."""
        mock_evaluator = MagicMock()
        mock_evaluator_cls.return_value = mock_evaluator

        mock_result = MagicMock()
        mock_result.strategy = "Random"
        mock_result.scenario = "HorizontalCR"
        mock_result.mean_reward = -10.0

        mock_evaluator.evaluate_random.return_value = mock_result
        mock_evaluator.evaluate_rule_based.return_value = mock_result
        mock_evaluator.evaluate_ppo.side_effect = FileNotFoundError("not found")
        mock_evaluator.evaluate_sac.side_effect = FileNotFoundError("not found")
        mock_evaluator.evaluate_td3.side_effect = FileNotFoundError("not found")
        mock_evaluator.evaluate_ddpg.side_effect = FileNotFoundError("not found")

        from scripts.evaluate_all import evaluate_all_scenarios
        results = evaluate_all_scenarios(
            scenario_name="HorizontalCR",
            num_aircraft=3,
            max_steps=50,
            num_episodes=5,
            model_dir=Path("nonexistent"),
        )

        # Should still return Random and RuleBased results
        assert len(results) >= 2
