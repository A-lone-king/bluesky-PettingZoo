"""Capacity violation penalty reward component."""

from __future__ import annotations

from typing import Any

from bluesky_pettingzoo.rewards.base import RewardComponent
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


class CapacityPenalty(RewardComponent):
    """Penalizes when the number of aircraft in a sector exceeds capacity.

    Returns ``(aircraft_count - max_aircraft) * penalty_per_excess`` when
    the count exceeds the configured maximum, otherwise 0.

    Config keys (under ``components.capacity``):
    - ``max_aircraft``: capacity threshold (default 5)
    - ``penalty_per_excess``: penalty per aircraft over the limit (default -10)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        comp = config.get("components", {}).get("capacity", {})
        self._max_aircraft: int = comp.get("max_aircraft", 5)
        self._penalty_per_excess: float = comp.get("penalty_per_excess", -10.0)

    def compute(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: DiscreteAction,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
    ) -> float:
        """Return penalty proportional to the number of excess aircraft."""
        count = len(all_states)
        excess = count - self._max_aircraft
        if excess <= 0:
            return 0.0
        return excess * self._penalty_per_excess

    def reset(self) -> None:
        """No internal state to reset."""
        pass
