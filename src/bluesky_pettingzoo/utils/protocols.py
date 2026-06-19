"""Protocol definitions for component interfaces.

Replaces hasattr() duck-typing with explicit structural typing.
Used by ObservationBuilder and BlueSkyMARLEnv for component detection.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from bluesky_pettingzoo.utils.types import AircraftState

# ---------------------------------------------------------------------------
# Reward component protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class EfficiencyComponent(Protocol):
    """Protocol for efficiency/delay reward components with goal tracking."""

    _goals: dict[str, tuple[float, float]]

    def set_goal(self, agent_id: str, lat: float, lon: float) -> None:
        """Set the goal waypoint for an agent."""
        ...


@runtime_checkable
class DelayComponent(Protocol):
    """Protocol for delay penalty component with expected steps tracking."""

    _expected_steps: dict[str, int]

    def set_goal(
        self,
        agent_id: str,
        distance_nm: float,
        speed_kt: float,
        dt: float,
    ) -> None:
        """Set the goal for delay computation."""
        ...


@runtime_checkable
class ConflictComponent(Protocol):
    """Protocol for conflict penalty component with conflict status query."""

    def get_conflict_status(
        self,
        own: AircraftState,
        others: list[AircraftState],
    ) -> str:
        """Return conflict level string for the agent."""
        ...


@runtime_checkable
class ObstacleComponent(Protocol):
    """Protocol for obstacle intrusion penalty component."""

    _obstacles: list[list[tuple[float, float]]]

    def set_obstacles(self, obstacles: list[list[tuple[float, float]]]) -> None:
        """Set obstacle polygons for the current episode."""
        ...

    def is_intruded(self, state: AircraftState) -> bool:
        """Check if an aircraft is inside any obstacle polygon."""
        ...


@runtime_checkable
class FlowEfficiencyComponent(Protocol):
    """Protocol for flow efficiency reward component."""

    def notify_sector_entry(self, agent_id: str, sector_id: str) -> None:
        """Record aircraft entering a sector."""
        ...


@runtime_checkable
class FairnessComponent(Protocol):
    """Protocol for fairness reward component with delay tracking."""

    _delays: dict[str, int]

    def set_delays(self, delays: dict[str, int]) -> None:
        """Set current delay values for all agents."""
        ...


# ---------------------------------------------------------------------------
# Scenario protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class PriorityScenario(Protocol):
    """Protocol for scenarios with priority-based entry ordering."""

    def get_priority(self, agent_id: str, state: AircraftState) -> float:
        """Return priority value for the agent (lower = higher priority)."""
        ...


@runtime_checkable
class DynamicEntryScenario(Protocol):
    """Protocol for scenarios supporting dynamic aircraft entry."""

    def get_initial_positions(self, num_aircraft: int, rng: object) -> list[dict[str, float]]:
        """Return initial positions for dynamically added aircraft."""
        ...


@runtime_checkable
class ObstacleScenario(Protocol):
    """Protocol for scenarios providing obstacle polygons."""

    def get_obstacles(self) -> list[list[tuple[float, float]]]:
        """Return list of obstacle polygons."""
        ...


@runtime_checkable
class NavigationScenario(Protocol):
    """Protocol for scenarios with LNAV navigation configuration."""

    def configure_npc_navigation(self, agent_id: str, wrapper: object) -> None:
        """Configure LNAV navigation for the agent."""
        ...


# ---------------------------------------------------------------------------
# Renderer protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class RendererDataSource(Protocol):
    """Protocol for renderer data source interface.

    Renderers should depend on this protocol instead of directly accessing
    environment internals (env.agents, env.pz_env). This decouples
    renderers from environment implementation details.
    """

    def get_aircraft_states(self) -> dict[str, Any]:
        """Return current aircraft states keyed by agent ID."""
        ...

    def get_waypoints(self) -> dict[str, dict[str, float]] | None:
        """Return goal waypoints keyed by agent ID."""
        ...

    def get_step_count(self) -> int:
        """Return current simulation step count."""
        ...

    def get_active_agents(self) -> list[str]:
        """Return list of active agent IDs."""
        ...


@runtime_checkable
class BoundedRenderer(Protocol):
    """Protocol for renderers that accept airspace bounds."""

    def set_bounds(self, bounds: dict[str, float]) -> None:
        """Set rendering bounds."""
        ...
