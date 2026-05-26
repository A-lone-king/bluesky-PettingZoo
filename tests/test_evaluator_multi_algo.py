"""Tests for ModelEvaluator multi-algorithm support."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bluesky_pettingzoo.training.evaluator import ModelEvaluator


class TestEvaluatorMultiAlgo:
    """Verify ModelEvaluator supports SAC, TD3, DDPG."""

    def test_has_evaluate_sac(self):
        assert hasattr(ModelEvaluator, "evaluate_sac")

    def test_has_evaluate_td3(self):
        assert hasattr(ModelEvaluator, "evaluate_td3")

    def test_has_evaluate_ddpg(self):
        assert hasattr(ModelEvaluator, "evaluate_ddpg")

    @patch("stable_baselines3.SAC")
    def test_evaluate_sac_loads_model(self, mock_sac_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = ([2, 2, 2], None)
        mock_sac_cls.load.return_value = mock_model

        env_factory = MagicMock()
        env = MagicMock()
        env.reset.return_value = ({}, {})
        env.step.return_value = ({}, 0.0, True, False, {})
        env.agents = ["AC000"]
        env.action_space = MagicMock()
        env.action_space.sample.return_value = [2, 2, 2]
        env_factory.return_value = env

        evaluator = ModelEvaluator(env_factory=env_factory, num_episodes=1, max_steps=5)
        result = evaluator.evaluate_sac(Path("test_model.zip"))

        mock_sac_cls.load.assert_called_once_with("test_model.zip")
        assert result.strategy == "SAC"

    @patch("stable_baselines3.TD3")
    def test_evaluate_td3_loads_model(self, mock_td3_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = ([2, 2, 2], None)
        mock_td3_cls.load.return_value = mock_model

        env_factory = MagicMock()
        env = MagicMock()
        env.reset.return_value = ({}, {})
        env.step.return_value = ({}, 0.0, True, False, {})
        env.agents = ["AC000"]
        env.action_space = MagicMock()
        env.action_space.sample.return_value = [2, 2, 2]
        env_factory.return_value = env

        evaluator = ModelEvaluator(env_factory=env_factory, num_episodes=1, max_steps=5)
        result = evaluator.evaluate_td3(Path("test_model.zip"))

        mock_td3_cls.load.assert_called_once_with("test_model.zip")
        assert result.strategy == "TD3"

    @patch("stable_baselines3.DDPG")
    def test_evaluate_ddpg_loads_model(self, mock_ddpg_cls):
        mock_model = MagicMock()
        mock_model.predict.return_value = ([2, 2, 2], None)
        mock_ddpg_cls.load.return_value = mock_model

        env_factory = MagicMock()
        env = MagicMock()
        env.reset.return_value = ({}, {})
        env.step.return_value = ({}, 0.0, True, False, {})
        env.agents = ["AC000"]
        env.action_space = MagicMock()
        env.action_space.sample.return_value = [2, 2, 2]
        env_factory.return_value = env

        evaluator = ModelEvaluator(env_factory=env_factory, num_episodes=1, max_steps=5)
        result = evaluator.evaluate_ddpg(Path("test_model.zip"))

        mock_ddpg_cls.load.assert_called_once_with("test_model.zip")
        assert result.strategy == "DDPG"
