"""Tests for training module public exports (A7)."""

from __future__ import annotations

import pytest


class TestTrainingImports:
    """All training classes should be importable from the training package."""

    def test_import_csv_logger(self) -> None:
        from bluesky_pettingzoo.training import CSVLoggerCallback
        assert CSVLoggerCallback is not None

    def test_import_checkpoint_manager(self) -> None:
        from bluesky_pettingzoo.training import CheckpointManager, CheckpointMeta
        assert CheckpointManager is not None
        assert CheckpointMeta is not None

    def test_import_evaluator(self) -> None:
        from bluesky_pettingzoo.training import ModelEvaluator, EvalResult
        assert ModelEvaluator is not None
        assert EvalResult is not None
