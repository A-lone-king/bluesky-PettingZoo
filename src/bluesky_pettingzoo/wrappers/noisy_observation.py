"""NoisyObservationWrapper — adds Gaussian noise to observations."""

from __future__ import annotations

from typing import Any

import numpy as np


class NoisyObservationWrapper:
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
        self.env = env
        self.noise_level = noise_level
        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Delegated properties
    # ------------------------------------------------------------------

    @property
    def agents(self) -> list[str]:
        return self.env.agents

    @property
    def possible_agents(self) -> list[str]:
        return self.env.possible_agents

    # ------------------------------------------------------------------
    # Delegated methods
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> Any:
        return self.env.observation_space(agent)

    def action_space(self, agent: str) -> Any:
        return self.env.action_space(agent)

    def close(self) -> None:
        self.env.close()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        observations, infos = self.env.reset(**kwargs)
        noisy_obs = {aid: self._add_noise(obs) for aid, obs in observations.items()}
        return noisy_obs, infos

    def step(
        self,
        actions: dict[str, Any],
    ) -> tuple[dict, dict, dict, dict, dict]:
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
                    value + self._rng.normal(0, self.noise_level, size=value.shape).astype(value.dtype)
                    if isinstance(value, np.ndarray)
                    else value
                )
                for key, value in observation.items()
            }
        if isinstance(observation, np.ndarray):
            return observation + self._rng.normal(0, self.noise_level, size=observation.shape).astype(observation.dtype)
        return observation
