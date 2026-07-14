"""Unit tests for multi-algorithm comparison framework."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bluesky_pettingzoo.training.multi_algo_comparison import (
    AlgoScenarioResult,
    ComparisonSummary,
    CONVERGENCE_THRESHOLDS,
    DEFAULT_ALGORITHMS,
    DEFAULT_SCENARIOS,
    DEFAULT_TIMESTEPS,
    MultiAlgoComparison,
)


class TestAlgoScenarioResult:
    """Tests for AlgoScenarioResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = AlgoScenarioResult(
            algorithm="PPO",
            scenario="HorizontalCR",
            initial_reward=-50.0,
            final_reward=-5.0,
            reward_history=[-50.0, -30.0, -10.0, -5.0],
            convergence_threshold=-10.0,
            converged=True,
            total_timesteps=500_000,
            seed=42,
        )
        assert result.algorithm == "PPO"
        assert result.scenario == "HorizontalCR"
        assert result.initial_reward == -50.0
        assert result.final_reward == -5.0
        assert result.converged is True
        assert len(result.reward_history) == 4

    def test_not_converged(self):
        """Test non-converged result."""
        result = AlgoScenarioResult(
            algorithm="DDPG",
            scenario="HorizontalCR",
            initial_reward=-50.0,
            final_reward=-15.0,
            reward_history=[-50.0, -15.0],
            convergence_threshold=-10.0,
            converged=False,
            total_timesteps=500_000,
            seed=42,
        )
        assert result.converged is False

    def test_to_dict(self):
        """Test serialization to dict."""
        result = AlgoScenarioResult(
            algorithm="SAC",
            scenario="VerticalCR",
            initial_reward=-20.0,
            final_reward=5.0,
            reward_history=[-20.0, 5.0],
            convergence_threshold=0.0,
            converged=True,
            total_timesteps=100_000,
            seed=123,
        )
        d = result.to_dict()
        assert d["algorithm"] == "SAC"
        assert d["scenario"] == "VerticalCR"
        assert d["final_reward"] == 5.0
        assert d["converged"] is True
        assert d["reward_history"] == [-20.0, 5.0]

    def test_json_serialization(self):
        """Test JSON serialization."""
        result = AlgoScenarioResult(
            algorithm="TD3",
            scenario="HorizontalCR",
            initial_reward=-30.0,
            final_reward=-8.0,
            reward_history=[-30.0, -8.0],
            convergence_threshold=-10.0,
            converged=True,
            total_timesteps=50_000,
            seed=42,
        )
        json_str = json.dumps(result.to_dict())
        d = json.loads(json_str)
        assert d["algorithm"] == "TD3"


class TestComparisonSummary:
    """Tests for ComparisonSummary dataclass."""

    def test_creation(self):
        """Test basic creation."""
        results = [
            AlgoScenarioResult(
                algorithm="PPO", scenario="HorizontalCR",
                initial_reward=-50.0, final_reward=-5.0,
                reward_history=[-50.0, -5.0],
                convergence_threshold=-10.0, converged=True,
                total_timesteps=500_000, seed=42,
            ),
            AlgoScenarioResult(
                algorithm="SAC", scenario="HorizontalCR",
                initial_reward=-50.0, final_reward=-15.0,
                reward_history=[-50.0, -15.0],
                convergence_threshold=-10.0, converged=False,
                total_timesteps=500_000, seed=42,
            ),
        ]
        summary = ComparisonSummary(
            results=results,
            scenarios=["HorizontalCR"],
            algorithms=["PPO", "SAC"],
            timestamp="2026-07-14",
            total_timesteps=500_000,
        )
        assert len(summary.results) == 2
        assert summary.all_converged is False

    def test_all_converged(self):
        """Test all_converged when all results pass."""
        results = [
            AlgoScenarioResult(
                algorithm="PPO", scenario="HorizontalCR",
                initial_reward=-50.0, final_reward=-5.0,
                reward_history=[-50.0, -5.0],
                convergence_threshold=-10.0, converged=True,
                total_timesteps=500_000, seed=42,
            ),
            AlgoScenarioResult(
                algorithm="SAC", scenario="VerticalCR",
                initial_reward=-20.0, final_reward=5.0,
                reward_history=[-20.0, 5.0],
                convergence_threshold=0.0, converged=True,
                total_timesteps=500_000, seed=42,
            ),
        ]
        summary = ComparisonSummary(
            results=results,
            scenarios=["HorizontalCR", "VerticalCR"],
            algorithms=["PPO", "SAC"],
            timestamp="2026-07-14",
            total_timesteps=500_000,
        )
        assert summary.all_converged is True

    def test_to_dict(self):
        """Test serialization."""
        results = [
            AlgoScenarioResult(
                algorithm="PPO", scenario="HorizontalCR",
                initial_reward=-50.0, final_reward=-5.0,
                reward_history=[-50.0, -5.0],
                convergence_threshold=-10.0, converged=True,
                total_timesteps=500_000, seed=42,
            ),
        ]
        summary = ComparisonSummary(
            results=results,
            scenarios=["HorizontalCR"],
            algorithms=["PPO"],
            timestamp="2026-07-14",
            total_timesteps=500_000,
        )
        d = summary.to_dict()
        assert d["all_converged"] is True
        assert len(d["results"]) == 1


class TestMultiAlgoComparison:
    """Tests for MultiAlgoComparison class."""

    def test_default_initialization(self):
        """Test default configuration."""
        comp = MultiAlgoComparison()
        assert comp.scenarios == DEFAULT_SCENARIOS
        assert comp.algorithms == DEFAULT_ALGORITHMS
        assert comp.total_timesteps == DEFAULT_TIMESTEPS
        assert comp.seed == 42

    def test_custom_initialization(self):
        """Test custom configuration."""
        comp = MultiAlgoComparison(
            scenarios=["HorizontalCR"],
            algorithms=["PPO", "SAC"],
            total_timesteps=100_000,
            seed=123,
        )
        assert comp.scenarios == ["HorizontalCR"]
        assert comp.algorithms == ["PPO", "SAC"]
        assert comp.total_timesteps == 100_000
        assert comp.seed == 123

    def test_convergence_thresholds(self):
        """Test that convergence thresholds are correct."""
        assert CONVERGENCE_THRESHOLDS["HorizontalCR"] == -10.0
        assert CONVERGENCE_THRESHOLDS["VerticalCR"] == 0.0

    def test_run_with_mock_train_fn(self):
        """Test run with a mock training function."""
        import tempfile
        import shutil
        tmp_dir = tempfile.mkdtemp()
        call_count = [0]

        def mock_train_fn(algorithm: str, scenario: str, timesteps: int, seed: int):
            call_count[0] += 1
            if scenario == "HorizontalCR":
                return -50.0, -5.0, [-50.0, -30.0, -10.0, -5.0]
            else:
                return -20.0, 5.0, [-20.0, -10.0, 0.0, 5.0]

        try:
            comp = MultiAlgoComparison(
                scenarios=["HorizontalCR", "VerticalCR"],
                algorithms=["PPO", "SAC"],
                total_timesteps=10_000,
                seed=42,
            )

            summary = comp.run(
                train_fn=mock_train_fn,
                save_results=True,
                results_dir=str(Path(tmp_dir) / "results"),
            )

            assert call_count[0] == 4
            assert len(summary.results) == 4
            assert summary.all_converged is True

            results_file = Path(tmp_dir) / "results" / "multi_algo_comparison.json"
            assert results_file.exists()

            with open(results_file, encoding="utf-8") as f:
                data = json.load(f)
            assert data["all_converged"] is True
            assert len(data["results"]) == 4
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_run_not_converged(self):
        """Test run when not all algorithms converge."""

        def mock_train_fn(algorithm: str, scenario: str, timesteps: int, seed: int):
            if algorithm == "DDPG" and scenario == "HorizontalCR":
                return -50.0, -15.0, [-50.0, -15.0]  # Not converged
            return -50.0, -5.0, [-50.0, -5.0]

        comp = MultiAlgoComparison(
            scenarios=["HorizontalCR"],
            algorithms=["PPO", "DDPG"],
            total_timesteps=10_000,
        )

        summary = comp.run(
            train_fn=mock_train_fn,
            save_results=False,
        )

        assert summary.all_converged is False
        ppo_result = next(r for r in summary.results if r.algorithm == "PPO")
        ddpg_result = next(r for r in summary.results if r.algorithm == "DDPG")
        assert ppo_result.converged is True
        assert ddpg_result.converged is False

    def test_generate_report(self):
        """Test report generation."""
        import tempfile
        import shutil
        tmp_dir = tempfile.mkdtemp()

        def mock_train_fn(algorithm: str, scenario: str, timesteps: int, seed: int):
            if scenario == "HorizontalCR":
                return -50.0, -5.0, [-50.0, -5.0]
            return -20.0, 5.0, [-20.0, 5.0]

        try:
            comp = MultiAlgoComparison(
                scenarios=["HorizontalCR", "VerticalCR"],
                algorithms=["PPO", "SAC"],
                total_timesteps=10_000,
            )

            summary = comp.run(
                train_fn=mock_train_fn,
                save_results=False,
            )

            report_path = str(Path(tmp_dir) / "report.md")
            report = comp.generate_report(summary, save_path=report_path)

            assert "多算法对比验证调参效果报告" in report
            assert "HorizontalCR" in report
            assert "VerticalCR" in report
            assert "PPO" in report
            assert "SAC" in report
            assert "收敛" in report

            assert Path(report_path).exists()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_generate_training_curve_data(self):
        """Test training curve data export."""
        import tempfile
        import shutil
        tmp_dir = tempfile.mkdtemp()

        def mock_train_fn(algorithm: str, scenario: str, timesteps: int, seed: int):
            return -50.0, -5.0, [-50.0, -30.0, -10.0, -5.0]

        try:
            comp = MultiAlgoComparison(
                scenarios=["HorizontalCR"],
                algorithms=["PPO"],
                total_timesteps=10_000,
            )

            summary = comp.run(
                train_fn=mock_train_fn,
                save_results=False,
            )

            curves_path = str(Path(tmp_dir) / "curves.json")
            comp.generate_training_curve_data(summary, save_path=curves_path)

            with open(curves_path, encoding="utf-8") as f:
                curves = json.load(f)

            assert "HorizontalCR" in curves
            assert "PPO" in curves["HorizontalCR"]
            assert len(curves["HorizontalCR"]["PPO"]) == 4
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_custom_convergence_thresholds(self):
        """Test custom convergence thresholds."""
        comp = MultiAlgoComparison(
            scenarios=["CustomScenario"],
            algorithms=["PPO"],
            convergence_thresholds={"CustomScenario": -5.0},
        )
        assert comp.thresholds["CustomScenario"] == -5.0
