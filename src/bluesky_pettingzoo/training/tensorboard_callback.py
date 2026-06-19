"""TensorBoard callback for stable-baselines3 training.

Logs episode rewards, lengths, and custom ATM metrics to TensorBoard.
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class TensorBoardCallback(BaseCallback):
    """Logs training metrics to TensorBoard.

    Tracks per-episode reward, length, conflict count, and arrival count.
    Writes to the TensorBoard log directory so ``tensorboard --logdir runs``
    shows reward curves during training.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self._episode = 0

    def _init_callback(self) -> None:
        """Verify TensorBoard is available."""
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except ImportError:
            if self.verbose > 0:
                print(
                    "Warning: tensorboard not installed. Install with: pip install tensorboard",
                )

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])

        for i, done in enumerate(dones):
            if not done:
                continue

            self._episode += 1
            info = infos[i] if i < len(infos) else {}
            ep_info = info.get("episode", {})

            reward = ep_info.get("r", rewards[i] if i < len(rewards) else 0.0)
            length = ep_info.get("l", 0)
            conflicts = info.get("conflicts", 0)
            arrivals = info.get("arrivals", 0)

            # Log to TensorBoard
            self.logger.record("episode/reward", float(reward))
            self.logger.record("episode/length", int(length))
            self.logger.record("episode/conflicts", int(conflicts))
            self.logger.record("episode/arrivals", int(arrivals))
            self.logger.record("episode/total", self._episode)

        return True
