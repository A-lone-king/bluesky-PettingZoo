"""Tests for CheckpointManager — model save/load/rotation (A3)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestCheckpointManager:
    """CheckpointManager should save, load, and rotate model checkpoints."""

    def _make_manager(self, tmp_path: Path, save_interval: int = 100, max_checkpoints: int = 3):
        from bluesky_pettingzoo.training.checkpoint import CheckpointManager

        return CheckpointManager(
            save_dir=tmp_path / "models",
            scenario="TestScenario",
            save_interval=save_interval,
            max_checkpoints=max_checkpoints,
        )

    def _mock_model(self) -> MagicMock:
        model = MagicMock()
        # Make save() actually create a .zip file
        def fake_save(path: str):
            Path(path).touch()
        model.save.side_effect = fake_save
        return model

    def test_save_creates_zip_and_json(self, tmp_path: Path) -> None:
        """maybe_save at interval boundary should create .zip and .json files."""
        mgr = self._make_manager(tmp_path, save_interval=100)
        model = self._mock_model()
        mgr.maybe_save(model, timestep=100, episode=1)

        save_dir = tmp_path / "models" / "TestScenario" / "PPO"
        zip_files = list(save_dir.glob("checkpoint_100.zip"))
        json_files = list(save_dir.glob("checkpoint_100.json"))
        assert len(zip_files) == 1
        assert len(json_files) == 1

    def test_save_skips_below_interval(self, tmp_path: Path) -> None:
        """maybe_save below interval should not create any files."""
        mgr = self._make_manager(tmp_path, save_interval=100)
        model = self._mock_model()
        mgr.maybe_save(model, timestep=50, episode=1)

        save_dir = tmp_path / "models" / "TestScenario" / "PPO"
        assert len(list(save_dir.glob("*.zip"))) == 0

    def test_save_final_creates_final_files(self, tmp_path: Path) -> None:
        """save_final should create _final.zip and _final.json."""
        mgr = self._make_manager(tmp_path)
        model = self._mock_model()
        mgr.save_final(model, timestep=500, episode=10)

        save_dir = tmp_path / "models" / "TestScenario" / "PPO"
        assert len(list(save_dir.glob("checkpoint_final.zip"))) == 1
        assert len(list(save_dir.glob("checkpoint_final.json"))) == 1

    def test_load_latest_returns_most_recent(self, tmp_path: Path) -> None:
        """load_latest should return the checkpoint with the highest timestep."""
        mgr = self._make_manager(tmp_path, save_interval=100)
        model = self._mock_model()
        mgr.maybe_save(model, timestep=100, episode=1)
        mgr.maybe_save(model, timestep=200, episode=2)
        mgr.maybe_save(model, timestep=300, episode=3)

        result = mgr.load_latest()
        assert result is not None
        path, meta = result
        assert meta.timestep == 300
        assert "checkpoint_300" in str(path)

    def test_load_latest_empty_dir_returns_none(self, tmp_path: Path) -> None:
        """load_latest on empty directory should return None."""
        mgr = self._make_manager(tmp_path)
        result = mgr.load_latest()
        assert result is None

    def test_fifo_rotation(self, tmp_path: Path) -> None:
        """When exceeding max_checkpoints, oldest checkpoints should be deleted."""
        mgr = self._make_manager(tmp_path, save_interval=100, max_checkpoints=2)
        model = self._mock_model()
        mgr.maybe_save(model, timestep=100, episode=1)
        mgr.maybe_save(model, timestep=200, episode=2)
        mgr.maybe_save(model, timestep=300, episode=3)

        save_dir = tmp_path / "models" / "TestScenario" / "PPO"
        zip_files = sorted(save_dir.glob("checkpoint_*.zip"))
        # Should only have 2 checkpoints (300 and 200), 100 should be deleted
        assert len(zip_files) == 2
        names = [f.name for f in zip_files]
        assert "checkpoint_300.zip" in names
        assert "checkpoint_200.zip" in names

    def test_json_metadata_correct(self, tmp_path: Path) -> None:
        """JSON metadata should contain all expected fields."""
        mgr = self._make_manager(tmp_path, save_interval=100)
        model = self._mock_model()
        mgr.maybe_save(model, timestep=100, episode=5)

        save_dir = tmp_path / "models" / "TestScenario" / "PPO"
        json_path = save_dir / "checkpoint_100.json"
        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["timestep"] == 100
        assert meta["episode"] == 5
        assert meta["scenario"] == "TestScenario"
        assert "seed" in meta
        assert "created_at" in meta
