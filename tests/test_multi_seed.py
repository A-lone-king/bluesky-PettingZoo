"""Unit tests for multi-seed training infrastructure."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bluesky_pettingzoo.training.multi_seed import MultiSeedTrainer, SeedResult, MultiSeedSummary


class TestSeedResult:
    """Tests for SeedResult dataclass."""

    def test_creation(self):
        """Test SeedResult creation with all fields."""
        result = SeedResult(
            seed=42,
            mean_reward=10.5,
            std_reward=2.3,
            mean_arrival_rate=0.8,
            mean_nmac_rate=0.1,
            mean_steps=35.0,
            model_path="models/HorizontalCR/PPO/seed_42/final_model.zip",
            eval_episodes=20,
        )
        assert result.seed == 42
        assert result.mean_reward == 10.5
        assert result.std_reward == 2.3
        assert result.mean_arrival_rate == 0.8
        assert result.mean_nmac_rate == 0.1
        assert result.mean_steps == 35.0
        assert result.model_path == "models/HorizontalCR/PPO/seed_42/final_model.zip"
        assert result.eval_episodes == 20

    def test_to_dict(self):
        """Test SeedResult conversion to dictionary."""
        result = SeedResult(
            seed=42,
            mean_reward=10.5,
            std_reward=2.3,
            mean_arrival_rate=0.8,
            mean_nmac_rate=0.1,
            mean_steps=35.0,
            model_path="models/test.zip",
            eval_episodes=20,
        )
        d = {
            "seed": 42,
            "mean_reward": 10.5,
            "std_reward": 2.3,
            "mean_arrival_rate": 0.8,
            "mean_nmac_rate": 0.1,
            "mean_steps": 35.0,
            "model_path": "models/test.zip",
            "eval_episodes": 20,
        }
        from dataclasses import asdict

        assert asdict(result) == d


class TestMultiSeedSummary:
    """Tests for MultiSeedSummary dataclass."""

    def test_creation(self):
        """Test MultiSeedSummary creation."""
        seed_results = [
            {"seed": 42, "mean_reward": 10.0, "std_reward": 1.0, "mean_arrival_rate": 0.8, "mean_nmac_rate": 0.1, "mean_steps": 30.0, "model_path": "models/seed_42.zip", "eval_episodes": 20},
            {"seed": 123, "mean_reward": 12.0, "std_reward": 1.5, "mean_arrival_rate": 0.85, "mean_nmac_rate": 0.05, "mean_steps": 28.0, "model_path": "models/seed_123.zip", "eval_episodes": 20},
        ]
        summary = MultiSeedSummary(
            scenario="HorizontalCR",
            algorithm="PPO",
            seeds=[42, 123],
            mean_reward=11.0,
            std_reward=1.0,
            mean_arrival_rate=0.825,
            std_arrival_rate=0.025,
            mean_nmac_rate=0.075,
            std_nmac_rate=0.025,
            mean_steps=29.0,
            std_steps=1.0,
            seed_results=seed_results,
            timestamp="2026-01-01T12:00:00",
        )
        assert summary.scenario == "HorizontalCR"
        assert summary.algorithm == "PPO"
        assert summary.seeds == [42, 123]
        assert summary.mean_reward == 11.0
        assert summary.seed_results == seed_results

    def test_json_serialization(self):
        """Test MultiSeedSummary JSON serialization."""
        seed_results = [
            {"seed": 42, "mean_reward": 10.0, "std_reward": 1.0, "mean_arrival_rate": 0.8, "mean_nmac_rate": 0.1, "mean_steps": 30.0, "model_path": "models/seed_42.zip", "eval_episodes": 20},
        ]
        summary = MultiSeedSummary(
            scenario="HorizontalCR",
            algorithm="PPO",
            seeds=[42],
            mean_reward=10.0,
            std_reward=1.0,
            mean_arrival_rate=0.8,
            std_arrival_rate=0.0,
            mean_nmac_rate=0.1,
            std_nmac_rate=0.0,
            mean_steps=30.0,
            std_steps=0.0,
            seed_results=seed_results,
            timestamp="2026-01-01T12:00:00",
        )
        from dataclasses import asdict

        data = asdict(summary)
        assert json.loads(json.dumps(data)) == data


class TestMultiSeedTrainer:
    """Tests for MultiSeedTrainer class."""

    def test_default_seeds(self):
        """Test DEFAULT_SEEDS constant."""
        assert MultiSeedTrainer.DEFAULT_SEEDS == [42, 123, 456, 789, 1024]

    def test_initialization_default_seeds(self):
        """Test trainer initialization with default seeds."""
        train_fn = MagicMock()
        trainer = MultiSeedTrainer(
            scenario="HorizontalCR",
            algorithm="PPO",
            train_fn=train_fn,
        )
        assert trainer.seeds == MultiSeedTrainer.DEFAULT_SEEDS
        assert trainer._scenario == "HorizontalCR"
        assert trainer._algorithm == "PPO"
        assert trainer._train_fn == train_fn
        assert trainer._save_dir == Path("models")
        assert trainer._results_dir == Path("results/multi_seed")

    def test_initialization_custom_seeds(self):
        """Test trainer initialization with custom seeds."""
        train_fn = MagicMock()
        trainer = MultiSeedTrainer(
            scenario="HorizontalCR",
            algorithm="PPO",
            train_fn=train_fn,
            seeds=[1, 2, 3],
            save_dir="custom_models",
            results_dir="custom_results",
        )
        assert trainer.seeds == [1, 2, 3]
        assert trainer._save_dir == Path("custom_models")
        assert trainer._results_dir == Path("custom_results")

    def test_get_seed_save_dir(self):
        """Test seed save directory generation."""
        train_fn = MagicMock()
        trainer = MultiSeedTrainer(
            scenario="HorizontalCR",
            algorithm="PPO",
            train_fn=train_fn,
            save_dir="models",
        )
        expected = Path("models") / "HorizontalCR" / "PPO" / "seed_42"
        assert trainer._get_seed_save_dir(42) == expected

    def test_train_all_with_mock(self, tmp_path):
        """Test train_all with mock train function."""
        mock_seed_results = [
            SeedResult(
                seed=42,
                mean_reward=10.0,
                std_reward=1.0,
                mean_arrival_rate=0.8,
                mean_nmac_rate=0.1,
                mean_steps=30.0,
                model_path=str(tmp_path / "seed_42" / "model.zip"),
                eval_episodes=20,
            ),
            SeedResult(
                seed=123,
                mean_reward=12.0,
                std_reward=1.5,
                mean_arrival_rate=0.85,
                mean_nmac_rate=0.05,
                mean_steps=28.0,
                model_path=str(tmp_path / "seed_123" / "model.zip"),
                eval_episodes=20,
            ),
        ]

        def mock_train_fn(seed: int, save_dir: Path) -> SeedResult:
            return mock_seed_results[0] if seed == 42 else mock_seed_results[1]

        trainer = MultiSeedTrainer(
            scenario="HorizontalCR",
            algorithm="PPO",
            train_fn=mock_train_fn,
            seeds=[42, 123],
            save_dir=str(tmp_path / "models"),
            results_dir=str(tmp_path / "results"),
        )

        summary = trainer.train_all()

        assert len(trainer.seed_results) == 2
        assert summary.scenario == "HorizontalCR"
        assert summary.algorithm == "PPO"
        assert summary.seeds == [42, 123]
        assert abs(summary.mean_reward - 11.0) < 0.01
        assert abs(summary.mean_arrival_rate - 0.825) < 0.001

    def test_aggregate_results_calculation(self):
        """Test aggregation calculation correctness."""
        seed_results = [
            SeedResult(seed=42, mean_reward=10.0, std_reward=1.0, mean_arrival_rate=0.8, mean_nmac_rate=0.1, mean_steps=30.0, model_path="m1", eval_episodes=20),
            SeedResult(seed=123, mean_reward=12.0, std_reward=1.5, mean_arrival_rate=0.85, mean_nmac_rate=0.05, mean_steps=28.0, model_path="m2", eval_episodes=20),
            SeedResult(seed=456, mean_reward=14.0, std_reward=2.0, mean_arrival_rate=0.9, mean_nmac_rate=0.02, mean_steps=26.0, model_path="m3", eval_episodes=20),
        ]

        rewards = np.array([r.mean_reward for r in seed_results])
        arrival_rates = np.array([r.mean_arrival_rate for r in seed_results])
        nmac_rates = np.array([r.mean_nmac_rate for r in seed_results])
        steps = np.array([r.mean_steps for r in seed_results])

        expected_mean_reward = float(np.mean(rewards))
        expected_std_reward = float(np.std(rewards))
        expected_mean_arrival = float(np.mean(arrival_rates))
        expected_std_arrival = float(np.std(arrival_rates))

        train_fn = MagicMock()
        trainer = MultiSeedTrainer("Test", "PPO", train_fn, seeds=[42, 123, 456])
        trainer._seed_results = seed_results

        summary = trainer._aggregate_results()

        assert abs(summary.mean_reward - expected_mean_reward) < 0.001
        assert abs(summary.std_reward - expected_std_reward) < 0.001
        assert abs(summary.mean_arrival_rate - expected_mean_arrival) < 0.001
        assert abs(summary.std_arrival_rate - expected_std_arrival) < 0.001

    def test_save_and_load_summary(self, tmp_path):
        """Test summary save and load round-trip."""
        seed_results = [
            {"seed": 42, "mean_reward": 10.0, "std_reward": 1.0, "mean_arrival_rate": 0.8, "mean_nmac_rate": 0.1, "mean_steps": 30.0, "model_path": "models/seed_42.zip", "eval_episodes": 20},
        ]
        summary = MultiSeedSummary(
            scenario="HorizontalCR",
            algorithm="PPO",
            seeds=[42],
            mean_reward=10.0,
            std_reward=1.0,
            mean_arrival_rate=0.8,
            std_arrival_rate=0.0,
            mean_nmac_rate=0.1,
            std_nmac_rate=0.0,
            mean_steps=30.0,
            std_steps=0.0,
            seed_results=seed_results,
            timestamp="2026-01-01T12:00:00",
        )

        results_dir = tmp_path / "results"
        trainer = MultiSeedTrainer("HorizontalCR", "PPO", MagicMock(), results_dir=str(results_dir))
        trainer._save_summary(summary)

        loaded = MultiSeedTrainer.load_summary("HorizontalCR", "PPO", results_dir=str(results_dir))
        assert loaded.scenario == summary.scenario
        assert loaded.algorithm == summary.algorithm
        assert loaded.seeds == summary.seeds
        assert loaded.mean_reward == summary.mean_reward
        assert loaded.mean_arrival_rate == summary.mean_arrival_rate

    def test_format_summary(self):
        """Test summary formatting."""
        seed_results = [
            {"seed": 42, "mean_reward": 10.0, "std_reward": 1.0, "mean_arrival_rate": 0.8, "mean_nmac_rate": 0.1, "mean_steps": 30.0, "model_path": "m1", "eval_episodes": 20},
            {"seed": 123, "mean_reward": 12.0, "std_reward": 1.5, "mean_arrival_rate": 0.85, "mean_nmac_rate": 0.05, "mean_steps": 28.0, "model_path": "m2", "eval_episodes": 20},
        ]
        summary = MultiSeedSummary(
            scenario="HorizontalCR",
            algorithm="PPO",
            seeds=[42, 123],
            mean_reward=11.0,
            std_reward=1.0,
            mean_arrival_rate=0.825,
            std_arrival_rate=0.025,
            mean_nmac_rate=0.075,
            std_nmac_rate=0.025,
            mean_steps=29.0,
            std_steps=1.0,
            seed_results=seed_results,
            timestamp="2026-01-01T12:00:00",
        )

        trainer = MultiSeedTrainer("HorizontalCR", "PPO", MagicMock())
        formatted = trainer.format_summary(summary)

        assert "Multi-Seed Summary" in formatted
        assert "HorizontalCR" in formatted
        assert "PPO" in formatted
        assert "Mean Reward" in formatted
        assert "Mean Arrival Rate" in formatted
        assert "Mean NMAC Rate" in formatted
        assert "42:" in formatted
        assert "123:" in formatted