"""Delay penalty reward component."""

from __future__ import annotations

from typing import Any

import numpy as np

from bluesky_pettingzoo.rewards.base import RewardComponent
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


class DelayPenalty(RewardComponent):
    """Penalizes aircraft that exceed their expected arrival time.

    Call :meth:`set_goal` for each agent with the initial distance to goal,
    cruise speed, and simulation dt.  The component computes the expected
    number of steps to reach the goal and penalizes proportionally when the
    agent is overdue.  Expected steps are dynamically adjusted based on
    current speed to avoid penalizing agents that slow down for separation.

    Config keys (under ``components.delay``):
    - ``delay_penalty_per_step``: penalty per step overdue (default -0.05)
    """

    component_name = "delay"
    config_keys = {
        "delay_penalty_per_step": ("_penalty_per_step", -0.05),
    }
    _stateful_attrs = ["_goals"]

    def __init__(self, config: dict[str, Any]) -> None:
        self._penalty_per_step: float = -0.05
        self._goals: dict[str, tuple[float, float, float]] = {}
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
        self._goals[agent_id] = (distance_nm, speed_kt, dt)

    def _compute_expected_steps(self, distance_nm: float, speed_kt: float, dt: float) -> int:
        """Compute expected steps to reach goal given distance and speed."""
        if speed_kt <= 0 or dt <= 0:
            return 0
        dist_per_step_nm = speed_kt * dt / 3600.0
        return max(1, int(distance_nm / dist_per_step_nm))

    def compute(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: DiscreteAction | list[Any] | np.ndarray,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
        step_count: int = 0,
    ) -> float:
        """Return delay penalty proportional to how overdue the agent is.

        Expected steps are dynamically adjusted based on current speed.
        If agent slows down, deadline extends; if speeds up, deadline shortens.
        """
        goal = self._goals.get(agent_id)
        if goal is None:
            return 0.0

        init_distance, init_speed, dt = goal

        # Compute expected steps at initial speed
        init_expected = self._compute_expected_steps(init_distance, init_speed, dt)

        # Get current speed from state
        current_speed = curr_state.tas

        # If stopped, no penalty (can't make progress)
        if current_speed <= 0:
            return 0.0

        # Dynamic adjustment: scale expected steps by speed ratio
        # If slower, deadline extends; if faster, deadline shortens
        if init_speed > 0:
            speed_ratio = current_speed / init_speed
            dynamic_expected = int(init_expected / speed_ratio)
        else:
            dynamic_expected = init_expected

        # Compute penalty
        overdue = step_count - dynamic_expected
        if overdue <= 0:
            return 0.0
        return overdue * self._penalty_per_step
