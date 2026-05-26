"""Tests for evaluate_baselines.py CLI and integration (A11)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestEvaluateScriptCLI:
    """Evaluate script should accept CLI arguments and output comparison tables."""

    def _parse_args(self, args: list[str]):
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.evaluate_baselines import parse_args
        return parse_args(args)

    def test_evaluate_accepts_scenario_arg(self) -> None:
        args = self._parse_args(["--scenario", "HorizontalCR"])
        assert args.scenario == "HorizontalCR"

    def test_evaluate_accepts_model_arg(self) -> None:
        args = self._parse_args(["--model", "models/checkpoint_final.zip"])
        assert args.model == "models/checkpoint_final.zip"

    def test_evaluate_accepts_episodes_arg(self) -> None:
        args = self._parse_args(["--episodes", "10"])
        assert args.episodes == 10

    def test_evaluate_accepts_seed_arg(self) -> None:
        args = self._parse_args(["--seed", "99"])
        assert args.seed == 99

    def test_evaluate_without_model_uses_random(self) -> None:
        args = self._parse_args(["--scenario", "HorizontalCR"])
        assert args.model is None

    def test_evaluate_output_contains_table(self, tmp_path: Path, capsys) -> None:
        """Running evaluate should print a table with strategy names."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.evaluate_baselines import run_evaluation

        args = MagicMock()
        args.scenario = "HorizontalCR"
        args.model = None
        args.episodes = 2
        args.seed = 42
        args.max_steps = 10
        args.num_aircraft = 2

        run_evaluation(args)
        captured = capsys.readouterr()
        assert "Random" in captured.out
