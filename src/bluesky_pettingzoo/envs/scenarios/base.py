"""Base scenario abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bluesky_pettingzoo.utils.types import (
    AircraftState,
    ConflictConfig,
    SpawnConfig,
)

_SCENARIO_REGISTRY: dict[str, str] = {
    "HorizontalCR": "HorizontalCRScenario",
    "VerticalCR": "VerticalCRScenario",
    "SectorCR": "SectorCRScenario",
    "PlanWaypoint": "PlanWaypointScenario",
    "Descent": "DescentScenario",
    "Merge": "MergeScenario",
    "RouteNav": "RouteNavScenario",
    "SectorCapacity": "SectorCapacityScenario",
    "StaticObstacle": "StaticObstacleScenario",
    "StarApproach": "StarApproachScenario",
    "WaypointNav": "WaypointNavScenario",
}

# Default conflict thresholds used by most scenarios
_DEFAULT_CONFLICT_CONFIG = ConflictConfig(
    nmac_horizontal_nm=5.0,
    nmac_vertical_ft=1000.0,
    warning_horizontal_nm=10.0,
    warning_vertical_ft=2000.0,
)


class BaseScenario(ABC):
    """Abstract base class for all scenarios.

    Subclasses must implement the 5 abstract methods.
    The 2 optional methods (update, reset) have default no-op implementations.

    Provides utility methods for common operations like agent ID generation
    and center point calculation.
    """

    @property
    def name(self) -> str:
        """Return scenario name for renderer selection.

        Derived from class name by stripping 'Scenario' suffix.
        e.g. 'HorizontalCRScenario' -> 'HorizontalCR'.
        """
        cls_name = type(self).__name__
        return cls_name.removesuffix("Scenario")

    @property
    def control_mode(self) -> str:
        """Return control mode: 'MULTI_RL' or 'SINGLE_RL'.

        MULTI_RL: all agents are controlled by RL.
        SINGLE_RL: only ego_agent is controlled; background agents follow presets.

        Returns:
            Control mode string. Default: 'MULTI_RL'.
        """
        return "MULTI_RL"

    @property
    def ego_agent(self) -> str | None:
        """Return the ego agent ID for SINGLE_RL mode.

        Returns:
            Ego agent ID string, or None if not in SINGLE_RL mode.
        """
        return None

    @property
    def background_agents(self) -> list[str]:
        """Return background agent IDs for SINGLE_RL mode.

        Returns:
            List of background agent IDs. Empty if not in SINGLE_RL mode.
        """
        return []

    @property
    def action_space_type(self) -> str:
        """Return action space type: 'discrete' or 'continuous'.

        Override in subclass to use continuous action space.

        Returns:
            Action space type string. Default: 'discrete'.
        """
        return getattr(self, "_action_space_type", "discrete")

    @action_space_type.setter
    def action_space_type(self, value: str) -> None:
        self._action_space_type = value

    @property
    def continuous_action_dims(self) -> int:
        """Return number of continuous action dimensions.

        Default: 3 (heading, altitude, speed).

        Returns:
            Number of continuous action dimensions.
        """
        return 3

    @staticmethod
    def generate_agent_ids(count: int, prefix: str = "AC") -> list[str]:
        """Generate agent IDs with sequential numbering.

        Args:
            count: Number of agents to generate.
            prefix: Agent ID prefix (default: "AC").

        Returns:
            List of agent ID strings like ["AC000", "AC001", ...].
        """
        return [f"{prefix}{i:03d}" for i in range(count)]

    @staticmethod
    def get_center_point(bounds: dict[str, float]) -> tuple[float, float]:
        """Calculate the center point of airspace bounds.

        Args:
            bounds: Dictionary with lat_min, lat_max, lon_min, lon_max.

        Returns:
            Tuple of (center_lat, center_lon).
        """
        center_lat = (bounds["lat_min"] + bounds["lat_max"]) / 2
        center_lon = (bounds["lon_min"] + bounds["lon_max"]) / 2
        return center_lat, center_lon

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

    def get_conflict_config(self) -> ConflictConfig:
        """Return conflict detection thresholds.

        Default implementation returns standard thresholds.
        Override in scenarios that need different values (e.g., MergeScenario).

        Returns:
            ConflictConfig with standard thresholds.
        """
        return _DEFAULT_CONFLICT_CONFIG

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

    def update_waypoint(
        self,
        agent_id: str,
        state: AircraftState,
    ) -> dict[str, float] | None:
        """Provide a new waypoint when the agent arrives at the current one.

        Override in scenarios that support waypoint streaming (continuous
        navigation). When the agent reaches the current waypoint, the
        environment calls this method instead of terminating the agent.

        Args:
            agent_id: Agent identifier.
            state: Current aircraft state at arrival.

        Returns:
            Dict with lat, lon keys for the next waypoint, or None to
            terminate the agent (default behavior).
        """
        return None

    def configure_npc_navigation(self, wrapper: Any) -> list[str]:
        """Configure NPC (background) aircraft navigation after setup.

        Override in scenarios that need background aircraft to follow
        real waypoints via BlueSky LNAV. Called once after setup().

        Args:
            wrapper: BlueSkyWrapper instance for sending commands.

        Returns:
            List of BlueSky commands sent (for logging/debugging).
        """
        return []

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

    def create_intruders(
        self,
        wrapper: Any,
        rng: np.random.RandomState | None = None,
    ) -> list[str]:
        """Create intruder aircraft using the wrapper's creconfs.

        Override in scenarios that use BlueSky's creconfs to generate
        conflict aircraft. Default returns empty list (no intruders).

        Args:
            wrapper: BlueSkyWrapper instance.
            rng: Random number generator for parameter variation.

        Returns:
            List of created intruder agent ID strings.
        """
        return []

    def get_background_actions(
        self,
        states: dict[str, Any],
    ) -> dict[str, list[int]]:
        """Return actions for background agents in SINGLE_RL mode.

        Override in scenarios that use SINGLE_RL mode to provide
        preset actions for background (non-controllable) agents.

        Args:
            states: Current aircraft states keyed by agent ID.

        Returns:
            Dict mapping background agent ID to action list.
            Default: empty dict (use no-op for all background agents).
        """
        return {}

    def get_initial_positions(self) -> dict[str, tuple[float, float]] | None:
        """Return initial (lat, lon) positions for each agent, or None to use random placement.

        Override this in scenarios that need aircraft spawned at specific locations
        (e.g. inside a polygon sector). When None is returned, the environment
        spawns aircraft at random positions within the airspace bounds.

        Returns:
            Dict mapping agent ID to (lat, lon) tuple, or None.
        """
        return None

    def get_priority(self, agent_id: str, state: AircraftState) -> float:
        """Return priority value for an agent (higher = higher priority).

        When multiple aircraft face a conflict, higher-priority aircraft
        maintain course while lower-priority aircraft yield.

        Args:
            agent_id: Agent identifier.
            state: Current aircraft state.

        Returns:
            Priority value, normalized to [-1, 1]. Default: 0.0 (equal).
        """
        return 0.0

    @property
    def num_aircraft_range(self) -> tuple[int, int] | None:
        """Return (min, max) range for dynamic aircraft count.

        Override in subclasses to enable procedural generation of varying
        aircraft counts.  When None, the fixed ``_num_aircraft`` value is
        used (default behavior).

        Returns:
            Tuple of (min, max) aircraft count, or None for fixed count.
        """
        return None

    def reset(self, rng: np.random.RandomState) -> None:
        """Reset scenario state and randomize parameters for next episode.

        Called before ``setup()`` on every episode.  Subclasses should
        clear stale internal state and optionally randomize mutable
        parameters (aircraft count, positions, etc.) using *rng*.

        The default implementation is a no-op for backward compatibility.

        Args:
            rng: Seeded random number generator for reproducibility.
        """

    @classmethod
    def from_config(cls, config_path: Path) -> BaseScenario:
        """Load a scenario instance from a YAML config file.

        The YAML must contain a ``scenario`` key mapping to a registered
        scenario name (e.g. ``HorizontalCR``).  All other keys are passed
        as keyword arguments to the scenario constructor.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            An instantiated scenario object.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the scenario name is not registered.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        scenario_name = data.pop("scenario")
        cls_name = _SCENARIO_REGISTRY.get(scenario_name)
        if cls_name is None:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. Registered: {list(_SCENARIO_REGISTRY.keys())}"
            )

        from bluesky_pettingzoo.envs.scenarios import (
            descent,
            horizontal_cr,
            merge,
            plan_waypoint,
            route_nav,
            sector_capacity,
            sector_cr,
            star_approach,
            static_obstacle,
            vertical_cr,
            waypoint_nav,
        )

        module_map = {
            "HorizontalCRScenario": horizontal_cr,
            "VerticalCRScenario": vertical_cr,
            "SectorCRScenario": sector_cr,
            "PlanWaypointScenario": plan_waypoint,
            "DescentScenario": descent,
            "MergeScenario": merge,
            "RouteNavScenario": route_nav,
            "SectorCapacityScenario": sector_capacity,
            "StaticObstacleScenario": static_obstacle,
            "StarApproachScenario": star_approach,
            "WaypointNavScenario": waypoint_nav,
        }
        mod = module_map[cls_name]
        scenario_cls = getattr(mod, cls_name)
        # Filter out keys not accepted by the constructor
        import inspect

        sig = inspect.signature(scenario_cls.__init__)
        valid_params = {
            p.name
            for p in sig.parameters.values()
            if p.name != "self"
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        filtered = {k: v for k, v in data.items() if k in valid_params}
        return scenario_cls(**filtered)  # type: ignore[no-any-return]
