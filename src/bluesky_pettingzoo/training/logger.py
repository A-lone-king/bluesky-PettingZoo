"""CSV training logger callback for stable-baselines3."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


class CSVLoggerCallback(BaseCallback):
    """Logs episode metrics to a CSV file during SB3 training.

    Writes one row per completed episode with columns:
    timestep, episode, reward, episode_length, conflicts, arrivals, timestamp.
    """

    def __init__(
        self,
        csv_path: Path,
        verbose: int = 0,
        algorithm: str = "PPO",
        action_space: str = "discrete",
    ) -> None:
        super().__init__(verbose=verbose)
        self._csv_path = Path(csv_path)
        self._file: Any = None
        self._writer: Any = None
        self._episode = 0
        self._algorithm = algorithm
        self._action_space = action_space

    def _init_callback(self) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "timestep",
                "episode",
                "reward",
                "episode_length",
                "conflicts",
                "arrivals",
                "algorithm",
                "action_space",
                "timestamp",
            ]
        )
        self._file.flush()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, done in enumerate(dones):
            if not done:
                continue

            self._episode += 1
            info = infos[i] if i < len(infos) else {}
            ep_info = info.get("episode", {})

            rewards_list = self.locals.get("rewards", [])
            default_r = rewards_list[i] if i < len(rewards_list) else 0.0
            reward = ep_info.get("r", default_r)
            length = ep_info.get("l", 0)
            conflicts = info.get("conflicts", 0)
            arrivals = info.get("arrivals", 0)

            self._writer.writerow(
                [
                    self.num_timesteps,
                    self._episode,
                    round(float(reward), 6),
                    int(length),
                    int(conflicts),
                    int(arrivals),
                    self._algorithm,
                    self._action_space,
                    round(time.time(), 3),
                ]
            )
            self._file.flush()

        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
