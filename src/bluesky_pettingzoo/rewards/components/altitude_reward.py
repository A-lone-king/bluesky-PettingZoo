"""Altitude reward component — regime-dependent altitude penalty.

Three regimes (inspired by bluesky-gym VerticalCR/Descent):
- Enroute: gentle penalty proportional to altitude error
- Near runway (within threshold): steep penalty for remaining altitude
- Crash (alt <= 0): fixed crash penalty
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np

from bluesky_pettingzoo.rewards.base import RewardComponent
from bluesky_pettingzoo.utils.geometry import haversine_distance
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


class AltitudeReward(RewardComponent):
    """Regime-dependent altitude penalty.

    Config keys (under components.altitude_reward):
        enroute_scale: penalty per ft of altitude error (default 5/3000)
        runway_scale: penalty per ft near runway (default 50/3000)
        runway_threshold_nm: distance threshold for runway regime (default 5.0)
        crash_penalty: fixed penalty when alt <= 0 (default -100.0)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        comp = config.get("components", {}).get("altitude_reward", {})
        self._enroute_scale: float = comp.get("enroute_scale", 5.0 / 3000)
        self._runway_scale: float = comp.get("runway_scale", 50.0 / 3000)
        self._runway_threshold_nm: float = comp.get("runway_threshold_nm", 5.0)
        self._crash_penalty: float = comp.get("crash_penalty", -100.0)
        self._goals: dict[str, tuple[float, float, float]] = {}  # (lat, lon, target_alt)

    def set_goal(self, agent_id: str, lat: float, lon: float, target_alt: float = 0.0) -> None:
        """Set the goal waypoint with target altitude for an agent."""
        self._goals[agent_id] = (lat, lon, target_alt)

    def compute(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: Union[DiscreteAction, list, np.ndarray],
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
        step_count: int = 0,
    ) -> float:
        goal = self._goals.get(agent_id)
        if goal is None:
            return 0.0

        # Crash check
        if curr_state.alt <= 0:
            return self._crash_penalty

        goal_lat, goal_lon, target_alt = goal
        remaining_alt = abs(curr_state.alt - target_alt)

        # Determine regime by distance to goal
        dist_nm = haversine_distance(curr_state.lat, curr_state.lon, goal_lat, goal_lon)

        if dist_nm < self._runway_threshold_nm:
            # Near runway: steep penalty for remaining altitude
            return -remaining_alt * self._runway_scale

        # Enroute: gentle penalty for altitude error
        return -remaining_alt * self._enroute_scale

    def reset(self) -> None:
        self._goals.clear()
