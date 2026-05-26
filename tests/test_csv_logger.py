"""Tests for CSVLoggerCallback — training log to CSV (A1)."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestCSVLoggerCallback:
    """CSVLoggerCallback should write episode metrics to a CSV file."""

    def _make_callback(self, csv_path: Path):
        from bluesky_pettingzoo.training.logger import CSVLoggerCallback

        return CSVLoggerCallback(csv_path=csv_path, verbose=0)

    def _mock_model(self):
        """Create a minimal mock model that the callback can attach to."""
        model = MagicMock()
        model.num_timesteps = 0
        return model

    def test_csv_file_created_on_init(self, tmp_path: Path) -> None:
        """After _init_callback, the CSV file should exist."""
        csv_path = tmp_path / "log.csv"
        cb = self._make_callback(csv_path)
        cb.init_callback(self._mock_model())
        assert csv_path.exists()

    def test_csv_headers_correct(self, tmp_path: Path) -> None:
        """CSV headers should include all expected columns."""
        csv_path = tmp_path / "log.csv"
        cb = self._make_callback(csv_path)
        cb.init_callback(self._mock_model())

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)

        expected = ["timestep", "episode", "reward", "episode_length", "conflicts", "arrivals", "algorithm", "action_space", "timestamp"]
        assert headers == expected

    def test_episode_logged_on_done(self, tmp_path: Path) -> None:
        """When an episode finishes (done=True), a row should be written."""
        csv_path = tmp_path / "log.csv"
        cb = self._make_callback(csv_path)
        model = self._mock_model()
        model.num_timesteps = 100
        cb.init_callback(model)

        # Simulate SB3 locals: one env, episode just finished
        cb.locals = {
            "dones": [True],
            "rewards": [5.0],
            "infos": [{"episode": {"r": 5.0, "l": 20}}],
        }
        cb.num_timesteps = 100
        cb._on_step()

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0][0] == "100"  # timestep
        assert rows[0][2] == "5.0"  # reward

    def test_multiple_episodes_appended(self, tmp_path: Path) -> None:
        """Multiple episodes should produce multiple rows."""
        csv_path = tmp_path / "log.csv"
        cb = self._make_callback(csv_path)
        cb.init_callback(self._mock_model())

        for i in range(3):
            cb.model.num_timesteps = (i + 1) * 50
            cb.num_timesteps = (i + 1) * 50
            cb.locals = {
                "dones": [True],
                "rewards": [float(i)],
                "infos": [{"episode": {"r": float(i), "l": 10 + i}}],
            }
            cb._on_step()

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)

        assert len(rows) == 3

    def test_file_closed_on_training_end(self, tmp_path: Path) -> None:
        """After _on_training_end, the file handle should be closed."""
        csv_path = tmp_path / "log.csv"
        cb = self._make_callback(csv_path)
        cb.init_callback(self._mock_model())
        cb._on_training_end()
        # Should be able to open and write to the file (no lock)
        with open(csv_path, encoding="utf-8") as f:
            f.read()

    def test_empty_training_no_error(self, tmp_path: Path) -> None:
        """Zero steps of training should not crash."""
        csv_path = tmp_path / "log.csv"
        cb = self._make_callback(csv_path)
        cb.init_callback(self._mock_model())
        cb._on_training_end()

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Only header, no data rows
        assert len(rows) == 1


class TestCSVLoggerAlgoColumns:
    """Verify CSV header includes algorithm and action_space columns."""

    def test_has_algorithm_column(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "log.csv"
        from bluesky_pettingzoo.training.logger import CSVLoggerCallback

        cb = CSVLoggerCallback(csv_path=csv_path, algorithm="PPO", action_space="discrete")
        cb.init_callback(self._mock_model())

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

        assert "algorithm" in header
        cb._on_training_end()

    def test_has_action_space_column(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "log.csv"
        from bluesky_pettingzoo.training.logger import CSVLoggerCallback

        cb = CSVLoggerCallback(csv_path=csv_path, algorithm="PPO", action_space="discrete")
        cb.init_callback(self._mock_model())

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

        assert "action_space" in header
        cb._on_training_end()

    def test_algorithm_value_in_row(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "log.csv"
        from bluesky_pettingzoo.training.logger import CSVLoggerCallback

        cb = CSVLoggerCallback(csv_path=csv_path, algorithm="SAC", action_space="continuous")
        model = self._mock_model()
        model.num_timesteps = 100
        cb.init_callback(model)
        cb.num_timesteps = 100
        cb.locals = {
            "dones": [True],
            "rewards": [10.0],
            "infos": [{"episode": {"r": 10.0, "l": 50}}],
        }
        cb._on_step()

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)

        algo_idx = header.index("algorithm")
        as_idx = header.index("action_space")
        assert row[algo_idx] == "SAC"
        assert row[as_idx] == "continuous"
        cb._on_training_end()

    def _mock_model(self):
        model = MagicMock()
        model.num_timesteps = 0
        return model
