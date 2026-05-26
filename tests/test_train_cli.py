"""Tests for training script CLI algorithm selection (spec4 F2).

Verify --algorithm and --action-space parameter parsing.
"""

from __future__ import annotations

import pytest

from scripts.train_ppo_scenarios import parse_args


class TestAlgorithmParam:
    """--algorithm parameter should parse correctly."""

    def test_default_algorithm(self) -> None:
        """Default algorithm should be PPO."""
        args = parse_args([])
        assert args.algorithm == "PPO"

    def test_ppo_algorithm(self) -> None:
        """--algorithm PPO should work."""
        args = parse_args(["--algorithm", "PPO"])
        assert args.algorithm == "PPO"

    def test_sac_algorithm(self) -> None:
        """--algorithm SAC should work."""
        args = parse_args(["--algorithm", "SAC"])
        assert args.algorithm == "SAC"

    def test_td3_algorithm(self) -> None:
        """--algorithm TD3 should work."""
        args = parse_args(["--algorithm", "TD3"])
        assert args.algorithm == "TD3"

    def test_ddpg_algorithm(self) -> None:
        """--algorithm DDPG should work."""
        args = parse_args(["--algorithm", "DDPG"])
        assert args.algorithm == "DDPG"

    def test_invalid_algorithm_raises(self) -> None:
        """Invalid algorithm should raise SystemExit."""
        with pytest.raises(SystemExit):
            parse_args(["--algorithm", "Invalid"])


class TestActionSpaceParam:
    """--action-space parameter should parse correctly."""

    def test_default_action_space(self) -> None:
        """Default action space should be discrete."""
        args = parse_args([])
        assert args.action_space == "discrete"

    def test_discrete_action_space(self) -> None:
        """--action-space discrete should work."""
        args = parse_args(["--action-space", "discrete"])
        assert args.action_space == "discrete"

    def test_continuous_action_space(self) -> None:
        """--action-space continuous should work."""
        args = parse_args(["--action-space", "continuous"])
        assert args.action_space == "continuous"

    def test_invalid_action_space_raises(self) -> None:
        """Invalid action space should raise SystemExit."""
        with pytest.raises(SystemExit):
            parse_args(["--action-space", "invalid"])
