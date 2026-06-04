"""WindFieldWrapper — injects wind field and augments observations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from gymnasium import spaces

try:
    import bluesky as bs
except ImportError:
    bs = None

from bluesky_pettingzoo.wrappers.base import EnvWrapperMixin

MAX_WIND = 50.0


class WindFieldWrapper(EnvWrapperMixin):
    """Wraps a ParallelEnv and injects a uniform wind field into BlueSky.

    Optionally augments each agent's observation with body-frame wind
    components (``wind_u``, ``wind_v``) so the RL policy can perceive
    headwind / crosswind.

    Args:
        env: The PettingZoo ParallelEnv to wrap.
        lat: Latitude of the wind field origin.
        lon: Longitude of the wind field origin.
        vnorth: North-component of wind velocity (knots).
        veast: East-component of wind velocity (knots).
        alt: Altitude for the wind point (None = surface).
        augment_obs: If True, add wind_u/wind_v to observations.
        seed: Random seed (reserved for future stochastic wind).
    """

    def __init__(
        self,
        env: Any,
        lat: float,
        lon: float,
        vnorth: float,
        veast: float,
        alt: float | None = None,
        augment_obs: bool = False,
        seed: int | None = None,
    ) -> None:
        self.lat = lat
        self.lon = lon
        self.vnorth = vnorth
        self.veast = veast
        self.alt = alt
        self.augment_obs = augment_obs
        self._rng = np.random.RandomState(seed)

        # Call EnvWrapperMixin.__init__ which sets self.env
        super().__init__(env)

        self._extended_space: spaces.Dict | None = None
        if self.augment_obs:
            base_space = self.env.observation_space(self.env.agents[0])
            self._extended_space = spaces.Dict(
                {
                    **base_space.spaces,
                    "wind_u": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                    "wind_v": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                }
            )
        else:
            pass

    # ------------------------------------------------------------------
    # Overridden methods
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Space[Any]:
        """Get observation space, with wind components if augmented."""
        if self._extended_space is not None:
            return self._extended_space
        return self.env.observation_space(agent)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        observations, infos = self.env.reset(**kwargs)
        if bs is None:
            raise ImportError(
                "bluesky is required for WindFieldWrapper. "
                "Install with: pip install bluesky-pettingzoo[bluesky]"
            )
        bs.traf.wind.addpointvne(self.lat, self.lon, self.vnorth, self.veast, self.alt)
        if self.augment_obs:
            observations = self._augment(observations)
        return observations, infos

    def step(
        self,
        actions: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        if self.augment_obs:
            observations = self._augment(observations)
        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _augment(self, observations: dict[str, Any]) -> dict[str, Any]:
        """Add wind_u/wind_v to every agent's observation."""
        augmented: dict[str, Any] = {}
        for agent_id, obs in observations.items():
            wind_u, wind_v = self._get_wind_observation(agent_id)
            augmented[agent_id] = {
                **obs,
                "wind_u": np.array([wind_u], dtype=np.float64),
                "wind_v": np.array([wind_v], dtype=np.float64),
            }
        return augmented

    def _get_wind_observation(self, agent_id: str) -> tuple[float, float]:
        """Get body-frame wind components for an agent, normalized by MAX_WIND."""
        states = self.env.unwrapped.aircraft_states
        state = states[agent_id]
        lat, lon, alt = state["lat"], state["lon"], state["alt"]
        hdg = state["hdg"]

        wind_n, wind_e = bs.traf.wind.getdata(lat, lon, alt)

        hdg_rad = math.radians(hdg)
        wind_u = (wind_n * math.cos(hdg_rad) + wind_e * math.sin(hdg_rad)) / MAX_WIND
        wind_v = (-wind_n * math.sin(hdg_rad) + wind_e * math.cos(hdg_rad)) / MAX_WIND

        return wind_u, wind_v
