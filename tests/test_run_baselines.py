"""Tests for run_baselines.py CLI and integration (A13)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock


class TestRunBaselinesCLI:
    """Run baselines script should accept CLI args and execute full pipeline."""

    def _parse_args(self, args: list[str]):
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.run_baselines import parse_args

        return parse_args(args)

    def test_run_baselines_accepts_scenario_arg(self) -> None:
        args = self._parse_args(["--scenario", "HorizontalCR"])
        assert args.scenario == "HorizontalCR"

    def test_run_baselines_accepts_timesteps_arg(self) -> None:
        args = self._parse_args(["--timesteps", "10000"])
        assert args.timesteps == 10000

    def test_run_baselines_accepts_episodes_arg(self) -> None:
        args = self._parse_args(["--episodes", "5"])
        assert args.episodes == 5

    def test_run_baselines_accepts_seed_arg(self) -> None:
        args = self._parse_args(["--seed", "99"])
        assert args.seed == 99

    def test_run_baselines_flow(self, tmp_path: Path, capsys) -> None:
        """Full pipeline: train → evaluate → output table."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.run_baselines import run_baselines

        args = MagicMock()
        args.scenario = "HorizontalCR"
        args.timesteps = 256
        args.episodes = 2
        args.seed = 42
        args.save_dir = str(tmp_path / "models")
        args.max_steps = 10
        args.num_aircraft = 2
        args.num_envs = 1
        args.device = "cpu"
        args.algorithm = "PPO"

        run_baselines(args)
        captured = capsys.readouterr()
        assert "Random" in captured.out
        assert "RuleBased" in captured.out
        assert "PPO" in captured.out
