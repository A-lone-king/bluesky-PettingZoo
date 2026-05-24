"""Flow scheduler for departure/arrival timing and sector handoff tracking."""

from __future__ import annotations

from typing import Any


class FlowScheduler:
    """Manages departure spacing, arrival spacing, and sector handoff tracking.

    Config keys (under ``flow``):
    - ``departure_interval``: minimum steps between consecutive departures (default 1)
    - ``arrival_interval``: minimum steps between consecutive arrivals (default 1)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        flow = config.get("flow", {})
        self._departure_interval: int = flow.get("departure_interval", 1)
        self._arrival_interval: int = flow.get("arrival_interval", 1)
        self._last_departure_step: int = -999
        self._last_arrival_step: int = -999
        self._handoff_counts: dict[str, int] = {}

    def check_departure(self, agent_id: str, step: int) -> bool:
        """Check if a departure is allowed at the given step.

        Args:
            agent_id: Agent identifier.
            step: Current simulation step.

        Returns:
            True if enough steps have passed since the last departure.
        """
        if step - self._last_departure_step >= self._departure_interval:
            self._last_departure_step = step
            return True
        return False

    def check_arrival(self, agent_id: str, step: int) -> bool:
        """Check if an arrival is allowed at the given step.

        Args:
            agent_id: Agent identifier.
            step: Current simulation step.

        Returns:
            True if enough steps have passed since the last arrival.
        """
        if step - self._last_arrival_step >= self._arrival_interval:
            self._last_arrival_step = step
            return True
        return False

    def notify_sector_change(
        self,
        agent_id: str,
        from_sector: str | None,
        to_sector: str | None,
    ) -> None:
        """Record a sector handoff for an aircraft.

        Args:
            agent_id: Agent identifier.
            from_sector: Previous sector (None if first entry).
            to_sector: New sector (None if leaving airspace).
        """
        self._handoff_counts[agent_id] = self._handoff_counts.get(agent_id, 0) + 1

    def get_handoff_delays(self) -> dict[str, int]:
        """Return per-agent handoff counts.

        Returns:
            Mapping of agent_id to number of sector handoffs.
        """
        return dict(self._handoff_counts)

    def reset(self) -> None:
        """Clear all state."""
        self._last_departure_step = -999
        self._last_arrival_step = -999
        self._handoff_counts.clear()
