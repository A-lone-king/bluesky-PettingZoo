"""NoisyObservationWrapper — adds Gaussian noise to observations."""

from __future__ import annotations

from typing import Any

import numpy as np

from bluesky_pettingzoo.wrappers.base import EnvWrapperMixin


class NoisyObservationWrapper(EnvWrapperMixin):
    """Wraps a ParallelEnv and adds Gaussian noise to observations.

    Noise is added independently to each agent's observation on every
    ``reset()`` and ``step()`` call.  Only ``ndarray`` values inside a
    Dict observation are perturbed; non-array values pass through unchanged.

    Args:
        env: The PettingZoo ParallelEnv to wrap.
        noise_level: Standard deviation of the Gaussian noise.
        seed: Random seed for reproducible noise generation.
    """

    def __init__(
        self,
        env: Any,
        noise_level: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.noise_level = noise_level
        self._rng = np.random.RandomState(seed)
        super().__init__(env)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the wrapped environment and add Gaussian noise to observations.

        Args:
            **kwargs: Arguments forwarded to the underlying env reset.

        Returns:
            Noisy observations and info dicts.
        """
        observations, infos = self.env.reset(**kwargs)
        noisy_obs = {aid: self._add_noise(obs) for aid, obs in observations.items()}
        return noisy_obs, infos

    def step(
        self,
        actions: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Step the wrapped environment and add Gaussian noise to observations.

        Args:
            actions: Actions for all agents.

        Returns:
            Noisy observations, rewards, terminations, truncations, and infos.
        """
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        noisy_obs = {aid: self._add_noise(obs) for aid, obs in observations.items()}
        return noisy_obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_noise(self, observation: Any) -> Any:
        """Add Gaussian noise to an observation dict."""
        if isinstance(observation, dict):
            return {
                key: (
                    value
                    + self._rng.normal(0, self.noise_level, size=value.shape).astype(value.dtype)  # type: ignore[attr-defined]
                    if isinstance(value, np.ndarray)
                    else value
                )
                for key, value in observation.items()
            }
        if isinstance(observation, np.ndarray):
            noise = self._rng.normal(0, self.noise_level, size=observation.shape)
            return observation + noise.astype(observation.dtype)  # type: ignore[attr-defined]
        return observation
