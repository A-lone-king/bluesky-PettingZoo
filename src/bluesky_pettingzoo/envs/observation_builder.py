"""Observation construction logic extracted from BlueSkyMARLEnv."""

from __future__ import annotations

from typing import Any

from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.utils.geometry import haversine_distance
from bluesky_pettingzoo.utils.types import AircraftState

# Conflict thresholds (defaults matching rewards.yaml)
_DEFAULT_NMAC_H = 5.0
_DEFAULT_NMAC_V = 1000.0
_DEFAULT_WARN_H = 10.0
_DEFAULT_WARN_V = 2000.0


class ObservationBuilder:
    """Builds observations and manages goal/waypoint lookups for BlueSkyMARLEnv."""

    def __init__(
        self,
        obs_manager: ObservationManager,
        reward_calculator: RewardCalculator,
        rewards_config: dict[str, Any],
        airspace: dict[str, float],
        airspace_cfg: dict[str, Any],
    ) -> None:
        self._obs_manager = obs_manager
        self._reward_calculator = reward_calculator
        self._airspace = airspace
        self._airspace_cfg = airspace_cfg

        # Conflict thresholds
        comp = rewards_config.get("components", {})
        thr = comp.get("conflict", {}).get("thresholds", {})
        self._nmac_h: float = thr.get("nmac_horizontal_nm", _DEFAULT_NMAC_H)
        self._nmac_v: float = thr.get("nmac_vertical_ft", _DEFAULT_NMAC_V)
        self._warn_h: float = thr.get("warning_horizontal_nm", _DEFAULT_WARN_H)
        self._warn_v: float = thr.get("warning_vertical_ft", _DEFAULT_WARN_V)

    # ------------------------------------------------------------------
    # Component finders
    # ------------------------------------------------------------------

    def _find_component(self, *attrs: str) -> Any:
        """Find the first reward component exposing all given attributes."""
        for comp, _ in self._reward_calculator.components:
            if all(hasattr(comp, a) for a in attrs):
                return comp
        return None

    def find_efficiency_component(self) -> Any:
        """Find the EfficiencyReward component (has set_goal + _goals)."""
        return self._find_component("set_goal", "_goals")

    def find_conflict_component(self) -> Any:
        """Find the ConflictPenalty component (has get_conflict_status)."""
        return self._find_component("get_conflict_status")

    def find_delay_component(self) -> Any:
        """Find the DelayPenalty component (has set_goal + _expected_steps)."""
        return self._find_component("set_goal", "_expected_steps")

    def find_obstacle_intrusion_component(self) -> Any:
        """Find the ObstacleIntrusion component (has set_obstacles)."""
        return self._find_component("set_obstacles")

    def find_flow_efficiency_component(self) -> Any:
        """Find the FlowEfficiencyReward component (has notify_sector_entry)."""
        return self._find_component("notify_sector_entry")

    def find_fairness_component(self) -> Any:
        """Find the FairnessReward component (has set_delays + _delays)."""
        return self._find_component("set_delays", "_delays")

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def build(
        self,
        all_states: dict[str, AircraftState],
        agents: list[str],
        scenario: Any = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build observations and infos for all agents.

        Args:
            all_states: Mapping of agent_id -> AircraftState for all aircraft.
            agents: List of active agent IDs.
            scenario: Optional scenario for priority computation.
        """
        observations: dict[str, Any] = {}
        infos: dict[str, Any] = {}
        obstacle_polygons = self._get_obstacle_polygons()

        # Compute priorities
        priorities: dict[str, float] = {}
        if scenario is not None and hasattr(scenario, "get_priority"):
            for aid, st in all_states.items():
                priorities[aid] = scenario.get_priority(aid, st)

        for agent_id in all_states:
            own = all_states.get(agent_id)
            if own is None:
                continue

            others = [s for aid, s in all_states.items() if aid != agent_id]
            conflict_status = self.compute_conflict_status(own, others)

            goal = self.make_goal(agent_id, own)
            result = self._obs_manager.generate(
                own_state=own,
                other_states=others,
                goal=goal,
                conflict_status=conflict_status,
                airspace=self._airspace_cfg,
                obstacle_polygons=obstacle_polygons,
                agent_priorities=priorities,
            )
            observations[agent_id] = result["observation"]
            infos[agent_id] = {
                "textual_state": result["textual_state"],
                "airspace_snapshot": result["airspace_snapshot"],
            }

        return observations, infos

    def make_goal(self, agent_id: str, own: AircraftState) -> dict[str, float]:
        """Get the goal for an agent, falling back to opposite-corner default."""
        eff = self.find_efficiency_component()
        if eff is not None and hasattr(eff, "_goals"):
            goal_tuple = eff._goals.get(agent_id)
            if goal_tuple is not None:
                return {"lat": goal_tuple[0], "lon": goal_tuple[1], "alt": own.alt, "hdg": own.hdg}
        mid_lat = (self._airspace["lat_min"] + self._airspace["lat_max"]) / 2
        mid_lon = (self._airspace["lon_min"] + self._airspace["lon_max"]) / 2
        return {
            "lat": self._airspace["lat_max"] if own.lat < mid_lat else self._airspace["lat_min"],
            "lon": self._airspace["lon_max"] if own.lon < mid_lon else self._airspace["lon_min"],
            "alt": own.alt,
            "hdg": own.hdg,
        }

    def get_waypoints_for_render(
        self,
        agents: list[str],
    ) -> dict[str, dict[str, float]] | None:
        """Get waypoints dict for the renderer."""
        eff = self.find_efficiency_component()
        if eff is None or not hasattr(eff, "_goals"):
            return None
        waypoints: dict[str, dict[str, float]] = {}
        for agent_id in agents:
            goal = eff._goals.get(agent_id)
            if goal is not None:
                waypoints[agent_id] = {
                    "lat": goal[0],
                    "lon": goal[1],
                    "alt": 35000.0,
                    "hdg": 0.0,
                }
        return waypoints if waypoints else None

    def default_observation(self) -> dict[str, Any]:
        """Return a random observation from the observation space."""
        return self._obs_manager.observation_space().sample()

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------

    def compute_conflict_status(
        self,
        own: AircraftState,
        others: list[AircraftState],
    ) -> str:
        """Compute conflict status using ConflictPenalty if available."""
        conflict_comp = self.find_conflict_component()
        if conflict_comp is not None:
            return conflict_comp.get_conflict_status(own, others)  # type: ignore[no-any-return]

        # Fallback: local implementation
        for other in others:
            h_dist = haversine_distance(own.lat, own.lon, other.lat, other.lon)
            v_dist = abs(own.alt - other.alt)
            if h_dist < self._nmac_h and v_dist < self._nmac_v:
                return "nmac"
            if h_dist < self._warn_h and v_dist < self._warn_v:
                return "warning"
        return "safe"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obstacle_polygons(self) -> list[list[tuple[float, float]]] | None:
        """Get obstacle polygons from the ObstacleIntrusion component."""
        comp = self.find_obstacle_intrusion_component()
        if comp is not None and hasattr(comp, "_obstacles"):
            return comp._obstacles  # type: ignore[no-any-return]
        return None
