"""SingleAgentGymWrapper — wraps ParallelEnv as a single-agent gymnasium.Env."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

_NOOP_ACTION = [2, 2, 2]


class SingleAgentGymWrapper(gymnasium.Env[Any, Any]):
    """Wraps a PettingZoo ParallelEnv as a single-agent gymnasium.Env.

    One "ego" agent is controlled by the RL policy. All other agents
    use a fixed noop policy ``[2, 2, 2]``.

    Args:
        env: The PettingZoo ParallelEnv to wrap.
        ego_agent: Agent ID controlled by the RL policy.
    """

    def __init__(self, env: Any, ego_agent: str = "AC000") -> None:
        self._env = env
        self._ego = ego_agent

        # Expose spaces for the ego agent
        self.observation_space: spaces.Space[Any] = env.observation_space(ego_agent)
        self.action_space: spaces.Space[Any] = env.action_space(ego_agent)

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the multi-agent environment and return the ego agent's observation.

        Args:
            seed: Optional random seed.
            options: Optional configuration overrides.

        Returns:
            Tuple of (observation, info) for the ego agent only.
        """
        observations, infos = self._env.reset(seed=seed, options=options)
        obs = observations[self._ego]
        info = infos.get(self._ego, {})
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Step with ego agent action; other agents receive noop actions.

        Args:
            action: Action for the ego agent.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Build full multi-agent action dict
        actions: dict[str, Any] = {}
        for agent_id in self._env.agents:
            if agent_id == self._ego:
                actions[agent_id] = action
            else:
                actions[agent_id] = list(_NOOP_ACTION)

        observations, rewards, terminations, truncations, infos = self._env.step(actions)

        obs = observations.get(self._ego, {})
        reward = float(rewards.get(self._ego, 0.0))
        terminated = bool(terminations.get(self._ego, False))
        truncated = bool(truncations.get(self._ego, False))
        info = infos.get(self._ego, {})

        # If ego was removed from agents mid-step, mark as terminated
        if self._ego not in self._env.agents and not terminated and not truncated:
            terminated = True

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the wrapped environment and release resources."""
        self._env.close()

    def render(self) -> None:
        """Render is not supported in single-agent mode (returns None)."""
        return None
