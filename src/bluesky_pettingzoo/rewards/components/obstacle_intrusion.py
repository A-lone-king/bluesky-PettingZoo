"""Obstacle intrusion penalty reward component.

Penalizes aircraft that enter restricted (no-fly) zones.
Each intrusion incurs a flat penalty and signals termination.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bluesky_pettingzoo.rewards.base import RewardComponent
from bluesky_pettingzoo.utils.geometry import point_in_polygon
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction

# Default penalty per intruded obstacle (matches bluesky-gym)
DEFAULT_INTRUSION_PENALTY = -5.0


class ObstacleIntrusion(RewardComponent):
    """Penalizes aircraft for entering restricted-area polygons.

    Obstacle polygons are set via :meth:`set_obstacles` (typically called
    by the scenario during ``reset``).  Each step, :meth:`compute` checks
    whether the aircraft's current position falls inside any polygon and
    returns the accumulated penalty.  Use :meth:`is_intruded` to query
    whether a specific agent is currently inside an obstacle.
    """

    def __init__(self, penalty: float = DEFAULT_INTRUSION_PENALTY) -> None:
        self._penalty = penalty
        # List of polygons, each a list of (lat, lon) tuples
        self._obstacles: list[list[tuple[float, float]]] = []

    def set_obstacles(self, obstacles: list[list[tuple[float, float]]]) -> None:
        """Set the obstacle polygons for the current episode.

        Args:
            obstacles: List of polygons. Each polygon is a list of
                (lat, lon) vertex tuples.
        """
        self._obstacles = obstacles

    def compute(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: DiscreteAction | list[Any] | np.ndarray,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
        step_count: int = 0,
    ) -> float:
        """Compute intrusion penalty for the agent.

        Returns the penalty multiplied by the number of obstacles
        the aircraft is currently inside (matches bluesky-gym behavior
        where penalty accumulates per intruded polygon).
        """
        intrusions = self._count_intrusions(curr_state)
        return self._penalty * intrusions

    def is_intruded(self, state: AircraftState) -> bool:
        """Check if an aircraft is inside any obstacle polygon.

        Args:
            state: Current aircraft state.

        Returns:
            True if the aircraft is inside at least one obstacle.
        """
        return self._count_intrusions(state) > 0

    def _count_intrusions(self, state: AircraftState) -> int:
        """Count how many obstacle polygons contain this aircraft."""
        count = 0
        for polygon in self._obstacles:
            if point_in_polygon(state.lat, state.lon, polygon):
                count += 1
        return count

    def reset(self) -> None:
        """Clear obstacle data for a new episode."""
        self._obstacles = []
