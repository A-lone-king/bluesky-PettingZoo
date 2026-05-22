"""Base scenario abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from bluesky_pettingzoo.utils.types import (
    AircraftState,
    ConflictConfig,
    SpawnConfig,
)


class BaseScenario(ABC):
    """Abstract base class for all scenarios.

    Subclasses must implement the 5 abstract methods.
    The 2 optional methods (update, reset) have default no-op implementations.
    """

    @abstractmethod
    def setup(
        self,
        rng: np.random.RandomState,
        airspace_bounds: dict[str, float],
    ) -> list[str]:
        """Initialize the scenario and return agent IDs.

        Args:
            rng: Seeded random number generator.
            airspace_bounds: Dict with lat_min, lat_max, lon_min, lon_max.

        Returns:
            List of agent ID strings to register.
        """

    @abstractmethod
    def get_spawn_config(self) -> SpawnConfig:
        """Return spawn parameters for aircraft."""

    @abstractmethod
    def get_conflict_config(self) -> ConflictConfig:
        """Return conflict detection thresholds."""

    def should_truncate(
        self,
        agent_id: str,
        state: AircraftState,
        airspace_bounds: dict[str, float],
    ) -> bool:
        """Check if an agent should be truncated (removed from episode).

        Default: truncate when aircraft leaves the rectangular airspace bounds.
        Override for custom boundaries (e.g. polygon sectors).

        Args:
            agent_id: Agent identifier.
            state: Current aircraft state.
            airspace_bounds: Dict with lat_min, lat_max, lon_min, lon_max.

        Returns:
            True if the agent should be truncated.
        """
        return (
            state.lat < airspace_bounds["lat_min"]
            or state.lat > airspace_bounds["lat_max"]
            or state.lon < airspace_bounds["lon_min"]
            or state.lon > airspace_bounds["lon_max"]
        )

    @abstractmethod
    def get_waypoint(self, agent_id: str) -> dict[str, float]:
        """Return the goal waypoint for an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            Dict with lat, lon, alt, hdg keys.
        """

    def update(
        self,
        step_count: int,
        all_states: dict[str, AircraftState],
    ) -> list[str]:
        """Called each step; return new agent IDs to add (default: none).

        Args:
            step_count: Current step number.
            all_states: All current aircraft states.

        Returns:
            List of new agent ID strings (empty by default).
        """
        return []

    def get_initial_positions(self) -> dict[str, tuple[float, float]] | None:
        """Return initial (lat, lon) positions for each agent, or None to use random placement.

        Override this in scenarios that need aircraft spawned at specific locations
        (e.g. inside a polygon sector). When None is returned, the environment
        spawns aircraft at random positions within the airspace bounds.

        Returns:
            Dict mapping agent ID to (lat, lon) tuple, or None.
        """
        return None

    def reset(self) -> None:
        """Reset scenario-internal state (default: no-op)."""
