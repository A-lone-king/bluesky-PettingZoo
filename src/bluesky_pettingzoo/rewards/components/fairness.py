"""Fairness reward component."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from bluesky_pettingzoo.rewards.base import RewardComponent
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


class FairnessReward(RewardComponent):
    """Penalizes unequal delay distribution across aircraft.

    When all aircraft have similar delays, the penalty is zero.  Greater
    inequality (measured by standard deviation of delays) produces a more
    negative reward.

    Call :meth:`set_delays` each step with per-agent delay values before
    calling :meth:`compute`.

    Config keys (under ``components.fairness``):
    - ``penalty_factor``: multiplier on the delay std dev (default 0.1)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        comp = config.get("components", {}).get("fairness", {})
        self._penalty_factor: float = comp.get("penalty_factor", 0.1)
        self._delays: dict[str, float] = {}

    def set_delays(self, delays: dict[str, float]) -> None:
        """Set per-agent delay values for this step.

        Args:
            delays: Mapping of agent_id to delay value (e.g. seconds overdue).
        """
        self._delays = dict(delays)

    def compute(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: DiscreteAction | list[Any] | np.ndarray,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
        step_count: int = 0,
    ) -> float:
        """Return fairness penalty based on delay inequality (std dev)."""
        if len(self._delays) < 2:
            return 0.0

        values = list(self._delays.values())
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)

        return -std * self._penalty_factor

    def reset(self) -> None:
        """Clear delay data."""
        self._delays.clear()
