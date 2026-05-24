"""Flow efficiency reward component."""

from __future__ import annotations

from typing import Any

from bluesky_pettingzoo.rewards.base import RewardComponent
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


class FlowEfficiencyReward(RewardComponent):
    """Rewards throughput: more aircraft passing through sectors per step → higher reward.

    Call :meth:`notify_sector_entry` each time an aircraft enters a sector.
    :meth:`compute` returns the total throughput count multiplied by the
    configured reward per aircraft.

    Config keys (under ``components.flow_efficiency``):
    - ``reward_per_aircraft``: reward per aircraft-sector entry (default 0.1)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        comp = config.get("components", {}).get("flow_efficiency", {})
        self._reward_per_aircraft: float = comp.get("reward_per_aircraft", 0.1)
        self._entries: list[tuple[str, str]] = []

    def notify_sector_entry(self, agent_id: str, sector_id: str) -> None:
        """Record that an aircraft entered a sector.

        Args:
            agent_id: Agent identifier.
            sector_id: Sector identifier.
        """
        self._entries.append((agent_id, sector_id))

    def compute(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: DiscreteAction,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
        step_count: int = 0,
    ) -> float:
        """Return flow efficiency reward based on total throughput this step."""
        return len(self._entries) * self._reward_per_aircraft

    def reset(self) -> None:
        """Clear all entry records."""
        self._entries.clear()
