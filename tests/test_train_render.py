"""Tests for --render CLI parameter in training script."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from train_ppo_scenarios import parse_args


class TestRenderArgument:
    """Verify --render argument parsing."""

    def test_render_flag_present(self):
        args = parse_args(["--scenario", "HorizontalCR", "--render"])
        assert args.render is True

    def test_render_default_false(self):
        args = parse_args(["--scenario", "HorizontalCR"])
        assert args.render is False

    def test_render_with_other_args(self):
        args = parse_args([
            "--scenario", "HorizontalCR",
            "--timesteps", "1000",
            "--render",
        ])
        assert args.render is True
        assert args.timesteps == 1000
