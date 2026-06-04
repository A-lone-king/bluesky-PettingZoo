"""Efficiency reward component: route deviation + arrival + step penalty."""

from __future__ import annotations

from typing import Any

import numpy as np

from bluesky_pettingzoo.rewards.base import RewardComponent
from bluesky_pettingzoo.utils.geometry import haversine_distance
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


class EfficiencyReward(RewardComponent):
    """Reward for route efficiency: deviation penalty, arrival reward, step cost.

    Components:
    - step_penalty: constant per-step cost
    - deviation_penalty: proportional to distance from goal, capped
    - arrival_reward: bonus when within arrival_threshold of goal
    """

    component_name = "efficiency"
    config_keys = {
        "max_deviation_nm": ("_max_deviation", 50.0),
        "deviation_penalty_scale": ("_deviation_scale", 0.0),
        "arrival_reward": ("_arrival_reward", 1.0),
        "step_penalty": ("_step_penalty", 0.0),
        "arrival_threshold_nm": ("_arrival_threshold", 2.0),
    }
    _stateful_attrs = ["_goals"]

    def __init__(self, config: dict[str, Any]) -> None:
        self._max_deviation: float = 50.0
        self._deviation_scale: float = 0.0
        self._arrival_reward: float = 1.0
        self._step_penalty: float = 0.0
        self._arrival_threshold: float = 2.0
        self._goals: dict[str, tuple[float, float]] = {}
        super().__init__(config)

    def set_goal(self, agent_id: str, lat: float, lon: float) -> None:
        """Set the goal waypoint for an agent."""
        self._goals[agent_id] = (lat, lon)

    def compute(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: DiscreteAction | list[Any] | np.ndarray,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
        step_count: int = 0,
    ) -> float:
        reward = self._step_penalty

        goal = self._goals.get(agent_id)
        if goal is None:
            return reward

        distance = haversine_distance(
            curr_state.lat,
            curr_state.lon,
            goal[0],
            goal[1],
        )

        deviation_penalty = -(distance / self._max_deviation) * self._deviation_scale
        reward += max(deviation_penalty, -self._deviation_scale)

        if distance < self._arrival_threshold:
            reward += self._arrival_reward

        return reward
