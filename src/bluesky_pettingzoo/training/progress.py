"""Progress bar callback for stable-baselines3 training."""

from __future__ import annotations

import sys
import time

from stable_baselines3.common.callbacks import BaseCallback


class ProgressCallback(BaseCallback):
    """Displays training progress with ETA and metrics.

    Shows a progress bar with percentage, elapsed time, ETA, and key metrics.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._start_time: float = 0.0
        self._last_print_len: int = 0
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []

    def _init_callback(self) -> None:
        self._start_time = time.time()
        self._total_timesteps = getattr(self.model, "total_timesteps", 0) or getattr(
            self.model, "_total_timesteps", 0
        )

    def _on_step(self) -> bool:
        # Track episode rewards
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, done in enumerate(dones):
            if not done:
                continue
            info = infos[i] if i < len(infos) else {}
            ep_info = info.get("episode", {})
            if ep_info:
                self._episode_rewards.append(ep_info.get("r", 0.0))
                self._episode_lengths.append(int(ep_info.get("l", 0)))

        # Print progress every 1000 steps
        if self.num_timesteps % 1000 == 0 or self.num_timesteps >= self._total_timesteps:
            self._print_progress()

        return True

    def _print_progress(self) -> None:
        """Print progress bar to stderr."""
        total = self._total_timesteps
        current = self.num_timesteps
        pct = current / total * 100

        elapsed = time.time() - self._start_time
        if current > 0:
            eta = elapsed / current * (total - current)
        else:
            eta = 0

        # Build progress bar
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = "█" * filled + "░" * (bar_len - filled)

        # Format time
        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta)

        # Recent metrics
        recent_rewards = self._episode_rewards[-10:] if self._episode_rewards else []
        recent_lengths = self._episode_lengths[-10:] if self._episode_lengths else []

        if recent_rewards:
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            avg_length = sum(recent_lengths) / len(recent_lengths)
            metrics_str = f" | avg_reward: {avg_reward:.2f} | avg_length: {avg_length:.1f}"
        else:
            metrics_str = ""

        # Clear previous line and print
        line = f"\r  [{bar}] {pct:5.1f}% | {elapsed_str} elapsed | ETA {eta_str}{metrics_str}"
        # Pad with spaces to clear previous line
        padding = " " * max(0, self._last_print_len - len(line))
        self._last_print_len = len(line)

        sys.stderr.write(line + padding)
        sys.stderr.flush()

    def _format_time(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS or MM:SS."""
        if seconds < 0:
            seconds = 0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _on_training_end(self) -> None:
        # Final newline after progress bar
        sys.stderr.write("\n")
        sys.stderr.flush()
