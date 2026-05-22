"""Delay penalty reward component."""

from __future__ import annotations

from typing import Any

from bluesky_pettingzoo.rewards.base import RewardComponent
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


class DelayPenalty(RewardComponent):
    """Penalizes each step to encourage agents to reach their goal quickly.

    Returns a fixed negative reward every step (default -0.01).
    The penalty value is configurable via ``components.delay.penalty``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        comp = config.get("components", {}).get("delay", {})
        self._penalty: float = comp.get("penalty", -0.01)

    def compute(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: DiscreteAction,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
    ) -> float:
        """Return fixed delay penalty."""
        return self._penalty

    def reset(self) -> None:
        """No internal state to reset."""
        pass
