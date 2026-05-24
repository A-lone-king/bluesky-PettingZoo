"""Efficiency reward component: route deviation + arrival + step penalty."""

from __future__ import annotations

from typing import Any

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

    def __init__(self, config: dict[str, Any]) -> None:
        comp = config.get("components", {}).get("efficiency", {})
        self._max_deviation: float = comp.get("max_deviation_nm", 50)
        self._deviation_scale: float = comp.get("deviation_penalty_scale", 5)
        self._arrival_reward: float = comp.get("arrival_reward", 10)
        self._step_penalty: float = comp.get("step_penalty", -0.01)
        self._arrival_threshold: float = comp.get("arrival_threshold_nm", 2)
        self._goals: dict[str, tuple[float, float]] = {}

    def set_goal(self, agent_id: str, lat: float, lon: float) -> None:
        """Set the goal waypoint for an agent."""
        self._goals[agent_id] = (lat, lon)

    def compute(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: DiscreteAction,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
        step_count: int = 0,
    ) -> float:
        reward = self._step_penalty

        goal = self._goals.get(agent_id)
        if goal is None:
            return reward

        distance = haversine_distance(
            curr_state.lat, curr_state.lon,
            goal[0], goal[1],
        )

        deviation_penalty = -(distance / self._max_deviation) * self._deviation_scale
        reward += max(deviation_penalty, -self._deviation_scale)

        if distance < self._arrival_threshold:
            reward += self._arrival_reward

        return reward

    def reset(self) -> None:
        self._goals.clear()
