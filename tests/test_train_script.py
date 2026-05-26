"""Tests for training script CLI parameter parsing (spec4 F1).

Verify that CLI arguments (--timesteps, --scenario, etc.) parse correctly.
"""

from __future__ import annotations

import pytest

from scripts.train_ppo_scenarios import parse_args


class TestParseArgs:
    """parse_args() should handle CLI arguments correctly."""

    def test_default_timesteps(self) -> None:
        """Default timesteps should be 500000."""
        args = parse_args([])
        assert args.timesteps == 500_000

    def test_custom_timesteps(self) -> None:
        """--timesteps should override default."""
        args = parse_args(["--timesteps", "500000"])
        assert args.timesteps == 500_000

    def test_default_scenario(self) -> None:
        """Default scenario should be HorizontalCR."""
        args = parse_args([])
        assert args.scenario == "HorizontalCR"

    def test_custom_scenario(self) -> None:
        """--scenario should override default."""
        args = parse_args(["--scenario", "VerticalCR"])
        assert args.scenario == "VerticalCR"

    def test_default_seed(self) -> None:
        """Default seed should be 42."""
        args = parse_args([])
        assert args.seed == 42

    def test_custom_seed(self) -> None:
        """--seed should override default."""
        args = parse_args(["--seed", "123"])
        assert args.seed == 123

    def test_default_max_steps(self) -> None:
        """Default max_steps should be 50."""
        args = parse_args([])
        assert args.max_steps == 50

    def test_custom_max_steps(self) -> None:
        """--max-steps should override default."""
        args = parse_args(["--max-steps", "100"])
        assert args.max_steps == 100

    def test_invalid_scenario_raises(self) -> None:
        """Invalid scenario should raise SystemExit."""
        with pytest.raises(SystemExit):
            parse_args(["--scenario", "InvalidScenario"])
