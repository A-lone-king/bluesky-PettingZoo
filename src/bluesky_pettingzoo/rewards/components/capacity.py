"""Capacity violation penalty reward component."""

from __future__ import annotations

import math
from typing import Any

from bluesky_pettingzoo.rewards.base import RewardComponent
from bluesky_pettingzoo.utils.geometry import assign_sector
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


class CapacityPenalty(RewardComponent):
    """Penalizes when aircraft count exceeds capacity.

    Supports two modes:

    **Global mode** (legacy): counts all aircraft against a single
    ``max_aircraft`` threshold.

    **Per-sector mode**: when ``sectors`` is present in the config,
    determines which sector each aircraft belongs to and penalizes
    based on per-sector capacity limits.

    Config keys (under ``components.capacity``):

    Global mode:
    - ``max_aircraft``: capacity threshold (default 5)
    - ``penalty_per_excess``: penalty per aircraft over limit (default -10)

    Per-sector mode:
    - ``sectors``: list of sector dicts with ``id``, ``bounds``/`polygon``,
      and ``capacity``
    - ``warning_threshold``: fraction of capacity at which warning starts
      (default 0.8)
    - ``penalty_per_excess``: penalty per aircraft over capacity (default -10)
    - ``warning_penalty``: penalty at warning level (default half of
      ``penalty_per_excess``)
    """

    component_name = "capacity"
    config_keys = {
        "max_aircraft": ("_max_aircraft", 5),
        "penalty_per_excess": ("_penalty_per_excess", -10.0),
        "sectors": ("_sectors", []),
        "warning_threshold": ("_warning_threshold", 0.8),
    }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        # Warning penalty depends on penalty_per_excess, so compute after base init
        comp = config.get("components", {}).get("capacity", {})
        self._warning_penalty: float = comp.get(
            "warning_penalty", self._penalty_per_excess / 2
        )

    def compute(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: DiscreteAction,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
        step_count: int = 0,
    ) -> float:
        """Return capacity penalty for the agent's sector."""
        if self._sectors:
            return self._compute_per_sector(agent_id, curr_state, all_states)
        return self._compute_global(all_states)

    def _compute_global(self, all_states: dict[str, AircraftState]) -> float:
        """Global capacity check (legacy mode)."""
        count = len(all_states)
        excess = count - self._max_aircraft
        if excess <= 0:
            return 0.0
        return excess * self._penalty_per_excess

    def _compute_per_sector(
        self,
        agent_id: str,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
    ) -> float:
        """Per-sector capacity check."""
        agent_sector = assign_sector(curr_state.lat, curr_state.lon, self._sectors)
        if agent_sector is None:
            return 0.0

        # Find capacity for agent's sector
        capacity = self._max_aircraft
        for s in self._sectors:
            if s["id"] == agent_sector:
                capacity = s.get("capacity", self._max_aircraft)
                break

        # Count aircraft in the same sector
        count = sum(
            1
            for state in all_states.values()
            if assign_sector(state.lat, state.lon, self._sectors) == agent_sector
        )

        excess = count - capacity
        if excess > 0:
            return excess * self._penalty_per_excess

        warning_limit = math.ceil(capacity * self._warning_threshold)
        if count >= warning_limit:
            return self._warning_penalty

        return 0.0
