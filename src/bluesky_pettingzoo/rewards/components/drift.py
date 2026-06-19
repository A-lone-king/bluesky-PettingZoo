"""Drift penalty reward component.

Computes penalty based on heading deviation from goal bearing.
Aligns with bluesky-gym's -0.1 * |drift_radians| pattern.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from bluesky_pettingzoo.rewards.base import RewardComponent
from bluesky_pettingzoo.utils.geometry import bearing
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


def _bound_angle_180(angle: float) -> float:
    """Bound angle to [-180, 180] degrees."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


class DriftPenalty(RewardComponent):
    """Penalty based on heading deviation from goal bearing.

    Computes: scale * |heading - bearing_to_goal| in radians.
    This provides a continuous gradient signal guiding the aircraft toward the goal.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        drift_cfg = config.get("components", {}).get("drift_penalty", {})
        self._scale: float = drift_cfg.get("scale", -0.1)
        self._goal_cache: dict[str, dict[str, float]] = {}

    def set_goal(self, agent_id: str, goal: dict[str, float]) -> None:
        """Cache goal position for an agent."""
        self._goal_cache[agent_id] = goal

    def compute(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: DiscreteAction | list[Any] | np.ndarray,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
        step_count: int = 0,
    ) -> float:
        """Compute drift penalty from heading deviation toward goal bearing.

        Args:
            agent_id: The agent identifier.
            prev_state: Previous aircraft state.
            action: Action taken.
            curr_state: Current aircraft state.
            all_states: All aircraft states.
            step_count: Current step number.

        Returns:
            Penalty proportional to absolute heading error in radians.
        """
        goal = self._goal_cache.get(agent_id)
        if goal is None:
            return 0.0

        # Compute bearing from current position to goal
        qdr = bearing(curr_state.lat, curr_state.lon, goal["lat"], goal["lon"])

        # Compute drift angle (heading - bearing) in degrees, bounded to [-180, 180]
        drift_deg = _bound_angle_180(curr_state.hdg - qdr)

        # Convert to radians and take absolute value
        drift_rad = abs(math.radians(drift_deg))

        return drift_rad * self._scale

    def reset(self) -> None:
        """Clear cached goal positions between episodes."""
        self._goal_cache.clear()
