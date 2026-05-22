"""Smoothness penalty reward component."""

from __future__ import annotations

from typing import Any

from bluesky_pettingzoo.rewards.base import RewardComponent
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction

_CENTER_IDX = 2


class SmoothnessPenalty(RewardComponent):
    """Penalizes non-zero action adjustments.

    Returns action_penalty if any axis has a non-zero adjustment,
    otherwise returns 0.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        comp = config.get("components", {}).get("smoothness", {})
        self._action_penalty: float = comp.get("action_penalty", -0.1)

    def compute(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: DiscreteAction,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
    ) -> float:
        """Return penalty if any action axis is non-zero."""
        if (
            action.heading_idx != _CENTER_IDX
            or action.altitude_idx != _CENTER_IDX
            or action.speed_idx != _CENTER_IDX
        ):
            return self._action_penalty
        return 0.0

    def reset(self) -> None:
        pass
