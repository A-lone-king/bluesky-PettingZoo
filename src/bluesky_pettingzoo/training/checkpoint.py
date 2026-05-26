"""Checkpoint manager for saving and loading PPO models."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CheckpointMeta:
    """Metadata for a saved checkpoint."""

    path: Path
    timestep: int
    episode: int
    scenario: str
    seed: int
    created_at: str


class CheckpointManager:
    """Manages PPO model checkpoint saving, loading, and rotation.

    Checkpoints are stored as .zip (model) + .json (metadata) pairs
    under ``save_dir/<scenario>/``.
    """

    def __init__(
        self,
        save_dir: Path,
        scenario: str,
        save_interval: int = 10_000,
        max_checkpoints: int = 5,
        seed: int = 42,
        algorithm: str = "PPO",
    ) -> None:
        self._save_dir = Path(save_dir) / scenario / algorithm
        self._scenario = scenario
        self._save_interval = save_interval
        self._max_checkpoints = max_checkpoints
        self._seed = seed

    def maybe_save(self, model: Any, timestep: int, episode: int) -> Path | None:
        """Save checkpoint if timestep is a multiple of save_interval.

        Returns the save path if saved, None otherwise.
        """
        if timestep % self._save_interval != 0 or timestep == 0:
            return None

        self._save_dir.mkdir(parents=True, exist_ok=True)
        name = f"checkpoint_{timestep}"
        zip_path = self._save_dir / f"{name}.zip"
        json_path = self._save_dir / f"{name}.json"

        model.save(str(zip_path))
        self._write_meta(json_path, timestep, episode)
        self._rotate()
        return zip_path

    def save_final(self, model: Any, timestep: int, episode: int) -> Path:
        """Save the final model checkpoint."""
        self._save_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self._save_dir / "checkpoint_final.zip"
        json_path = self._save_dir / "checkpoint_final.json"

        model.save(str(zip_path))
        self._write_meta(json_path, timestep, episode)
        return zip_path

    def load_latest(self) -> tuple[Path, CheckpointMeta] | None:
        """Load the most recent checkpoint.

        Returns (model_path, metadata) or None if no checkpoints exist.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda m: m.timestep)
        return latest.path, latest

    def list_checkpoints(self) -> list[CheckpointMeta]:
        """List all saved checkpoints (excluding _final)."""
        if not self._save_dir.exists():
            return []

        result: list[CheckpointMeta] = []
        for json_path in sorted(self._save_dir.glob("checkpoint_*.json")):
            if "_final" in json_path.name:
                continue
            try:
                meta = self._read_meta(json_path)
                if meta is not None:
                    result.append(meta)
            except (json.JSONDecodeError, KeyError):
                continue
        return result

    def _write_meta(self, json_path: Path, timestep: int, episode: int) -> None:
        meta = {
            "timestep": timestep,
            "episode": episode,
            "scenario": self._scenario,
            "seed": self._seed,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _read_meta(self, json_path: Path) -> CheckpointMeta | None:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        zip_path = json_path.with_suffix(".zip")
        return CheckpointMeta(
            path=zip_path,
            timestep=data["timestep"],
            episode=data["episode"],
            scenario=data["scenario"],
            seed=data["seed"],
            created_at=data["created_at"],
        )

    def _rotate(self) -> None:
        """Delete oldest checkpoints exceeding max_checkpoints."""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= self._max_checkpoints:
            return

        sorted_cps = sorted(checkpoints, key=lambda m: m.timestep)
        to_remove = sorted_cps[: len(sorted_cps) - self._max_checkpoints]
        for cp in to_remove:
            cp.path.unlink(missing_ok=True)
            cp.path.with_suffix(".json").unlink(missing_ok=True)
