"""FrameStackWrapper — stacks observations from the last N timesteps."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any

import numpy as np
from gymnasium import spaces

from bluesky_pettingzoo.wrappers.base import EnvWrapperMixin


class FrameStackWrapper(EnvWrapperMixin):
    """Wraps a ParallelEnv and stacks observations from the last N timesteps.

    For each agent, maintains a rolling buffer of the last ``stack_size``
    observations. The observation space is modified to include the stacked
    dimensions.

    Args:
        env: The PettingZoo ParallelEnv to wrap.
        stack_size: Number of frames to stack (must be >= 1).
        padding_type: Padding type for initial frames. Options:
            - "zero": Use zero arrays for padding
            - "repeat": Repeat the first observation for padding
    """

    def __init__(
        self,
        env: Any,
        stack_size: int = 4,
        padding_type: str = "zero",
    ) -> None:
        if stack_size < 1:
            raise ValueError(f"stack_size must be >= 1, got {stack_size}")
        if padding_type not in ("zero", "repeat"):
            raise ValueError(
                f"padding_type must be 'zero' or 'repeat', got {padding_type!r}"
            )

        self._stack_size = stack_size
        self._padding_type = padding_type
        self._obs_buffers: dict[str, deque[dict[str, Any]]] = {}
        self._stacked_obs_spaces: dict[str, spaces.Dict] = {}

        super().__init__(env)

        # Pre-compute stacked observation spaces for each agent
        for agent in self.possible_agents:
            orig_space = self.env.observation_space(agent)
            self._stacked_obs_spaces[agent] = self._build_stacked_space(
                orig_space, stack_size
            )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Dict:
        """Return the stacked observation space for an agent."""
        return self._stacked_obs_spaces[agent]

    def reset(
        self, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment and initialize frame buffers."""
        observations, infos = self.env.reset(**kwargs)

        # Initialize buffers for each agent
        self._obs_buffers = {}
        for agent_id, obs in observations.items():
            buffer: deque[dict[str, Any]] = deque(
                maxlen=self._stack_size
            )
            # Create padding observation
            if self._padding_type == "zero":
                padding = self._create_zero_observation(
                    self.env.observation_space(agent_id)
                )
            else:  # repeat
                padding = deepcopy(obs)

            # Fill buffer with padding observations
            for _ in range(self._stack_size - 1):
                buffer.append(deepcopy(padding))
            buffer.append(obs)

            self._obs_buffers[agent_id] = buffer

        # Stack observations
        stacked_obs = {
            aid: self._stack_obs(buf)
            for aid, buf in self._obs_buffers.items()
        }
        return stacked_obs, infos

    def step(
        self,
        actions: dict[str, Any],
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        """Step through environment and update frame stacks."""
        observations, rewards, terminations, truncations, infos = self.env.step(
            actions
        )

        # Update buffers
        for agent_id, obs in observations.items():
            if agent_id not in self._obs_buffers:
                # New agent (dynamic entry) — initialize buffer
                buffer: deque[dict[str, Any]] = deque(
                    maxlen=self._stack_size
                )
                if self._padding_type == "zero":
                    padding = self._create_zero_observation(
                        self.env.observation_space(agent_id)
                    )
                else:
                    padding = deepcopy(obs)
                for _ in range(self._stack_size - 1):
                    buffer.append(deepcopy(padding))
                self._obs_buffers[agent_id] = buffer

            self._obs_buffers[agent_id].append(obs)

        # Stack observations
        stacked_obs = {
            aid: self._stack_obs(buf)
            for aid, buf in self._obs_buffers.items()
            if aid in observations  # Only return obs for active agents
        }

        # Clean up terminated agents
        for agent_id in list(self._obs_buffers.keys()):
            if agent_id in terminations and terminations[agent_id]:
                del self._obs_buffers[agent_id]

        return stacked_obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stack_obs(self, buffer: deque[dict[str, Any]]) -> dict[str, Any]:
        """Stack observations from buffer into a single dict."""
        if not buffer:
            return {}

        # Get the keys from the first observation
        first_obs = buffer[0]
        stacked: dict[str, Any] = {}

        for key in first_obs:
            values = [obs[key] for obs in buffer if key in obs]

            if not values:
                continue

            first_val = values[0]
            if isinstance(first_val, np.ndarray):
                # Stack along new axis (stack_size, ...)
                stacked[key] = np.stack(values, axis=0)
            elif isinstance(first_val, dict):
                # Recursively stack nested dicts
                stacked[key] = self._stack_nested_dict(values)
            else:
                # For scalar or other types, keep as list or take last value
                stacked[key] = values[-1]

        return stacked

    def _stack_nested_dict(
        self, dicts: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Stack values from multiple dicts."""
        if not dicts:
            return {}

        result: dict[str, Any] = {}
        for key in dicts[0]:
            values = [d[key] for d in dicts if key in d]
            if not values:
                continue

            first_val = values[0]
            if isinstance(first_val, np.ndarray):
                result[key] = np.stack(values, axis=0)
            elif isinstance(first_val, dict):
                result[key] = self._stack_nested_dict(values)
            else:
                result[key] = values[-1]

        return result

    def _build_stacked_space(
        self, orig_space: spaces.Dict, stack_size: int
    ) -> spaces.Dict:
        """Build a new Dict space with stacked dimensions."""
        new_spaces: dict[str, spaces.Space[Any]] = {}

        for key, subspace in orig_space.spaces.items():
            if isinstance(subspace, spaces.Box):
                # Add stack_size dimension at the beginning
                new_shape = (stack_size,) + subspace.shape
                new_low = np.tile(
                    subspace.low, (stack_size,) + (1,) * len(subspace.shape)
                )
                new_high = np.tile(
                    subspace.high, (stack_size,) + (1,) * len(subspace.shape)
                )
                new_spaces[key] = spaces.Box(
                    low=new_low,
                    high=new_high,
                    shape=new_shape,
                    dtype=np.float32,
                )
            elif isinstance(subspace, spaces.Dict):
                # Recursively handle nested dicts
                new_spaces[key] = self._build_stacked_dict_space(
                    subspace, stack_size
                )
            elif isinstance(subspace, spaces.Discrete):
                # Stack discrete spaces as multi-discrete
                new_spaces[key] = spaces.MultiDiscrete(
                    [int(subspace.n)] * stack_size
                )
            else:
                # For other space types, keep as-is (not stacked)
                new_spaces[key] = subspace

        return spaces.Dict(new_spaces)

    def _build_stacked_dict_space(
        self, orig_space: spaces.Dict, stack_size: int
    ) -> spaces.Dict:
        """Build a stacked Dict space for nested observations."""
        new_spaces: dict[str, spaces.Space[Any]] = {}

        for key, subspace in orig_space.spaces.items():
            if isinstance(subspace, spaces.Box):
                new_shape = (stack_size,) + subspace.shape
                new_low = np.tile(
                    subspace.low, (stack_size,) + (1,) * len(subspace.shape)
                )
                new_high = np.tile(
                    subspace.high, (stack_size,) + (1,) * len(subspace.shape)
                )
                new_spaces[key] = spaces.Box(
                    low=new_low,
                    high=new_high,
                    shape=new_shape,
                    dtype=np.float32,
                )
            elif isinstance(subspace, spaces.Dict):
                new_spaces[key] = self._build_stacked_dict_space(
                    subspace, stack_size
                )
            else:
                new_spaces[key] = subspace

        return spaces.Dict(new_spaces)

    def _create_zero_observation(self, space: spaces.Dict) -> dict[str, Any]:
        """Create a zero-valued observation matching the space."""
        obs: dict[str, Any] = {}
        for key, subspace in space.spaces.items():
            if isinstance(subspace, spaces.Box):
                obs[key] = np.zeros(subspace.shape, dtype=subspace.dtype)
            elif isinstance(subspace, spaces.Dict):
                obs[key] = self._create_zero_observation(subspace)
            elif isinstance(subspace, spaces.Discrete):
                obs[key] = 0
            elif isinstance(subspace, spaces.MultiDiscrete):
                obs[key] = np.zeros(subspace.shape, dtype=subspace.dtype)
            elif isinstance(subspace, spaces.MultiBinary):
                obs[key] = np.zeros(subspace.shape, dtype=np.int8)
            else:
                # Fallback: try to create zero array
                try:
                    obs[key] = np.zeros(subspace.shape, dtype=subspace.dtype)
                except Exception:
                    obs[key] = 0
        return obs
