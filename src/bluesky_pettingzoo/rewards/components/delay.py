"""Delay penalty reward component."""

from __future__ import annotations

from typing import Any

from bluesky_pettingzoo.rewards.base import RewardComponent
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


class DelayPenalty(RewardComponent):
    """Penalizes aircraft that exceed their expected arrival time.

    Call :meth:`set_goal` for each agent with the initial distance to goal,
    cruise speed, and simulation dt.  The component computes the expected
    number of steps to reach the goal and penalizes proportionally when the
    agent is overdue.

    Config keys (under ``components.delay``):
    - ``delay_penalty_per_step``: penalty per step overdue (default -0.05)
    """

    component_name = "delay"
    config_keys = {
        "delay_penalty_per_step": ("_penalty_per_step", -0.05),
    }
    _stateful_attrs = ["_expected_steps"]

    def __init__(self, config: dict[str, Any]) -> None:
        self._expected_steps: dict[str, int] = {}
        super().__init__(config)

    def set_goal(
        self,
        agent_id: str,
        distance_nm: float,
        speed_kt: float,
        dt: float,
    ) -> None:
        """Set the goal for an agent and compute expected arrival steps.

        Args:
            agent_id: Agent identifier.
            distance_nm: Initial distance to goal in nautical miles.
            speed_kt: Cruise speed in knots.
            dt: Simulation time step in seconds.
        """
        if speed_kt <= 0 or dt <= 0:
            self._expected_steps[agent_id] = 0
            return
        dist_per_step_nm = speed_kt * dt / 3600.0
        self._expected_steps[agent_id] = max(1, int(distance_nm / dist_per_step_nm))

    def compute(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: DiscreteAction,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
        step_count: int = 0,
    ) -> float:
        """Return delay penalty proportional to how overdue the agent is."""
        expected = self._expected_steps.get(agent_id)
        if expected is None:
            return 0.0
        overdue = step_count - expected
        if overdue <= 0:
            return 0.0
        return overdue * self._penalty_per_step
