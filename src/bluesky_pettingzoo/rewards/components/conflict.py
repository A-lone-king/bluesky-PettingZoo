"""Conflict penalty reward component."""

from __future__ import annotations

from typing import Any

from bluesky_pettingzoo.rewards.base import RewardComponent
from bluesky_pettingzoo.utils.geometry import haversine_distance, project_position
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


class ConflictPenalty(RewardComponent):
    """Penalizes aircraft conflicts based on severity levels.

    Severity hierarchy (most severe first):
    1. NMAC: horizontal < nmac_h AND vertical < nmac_v
    2. Warning: horizontal < warn_h AND vertical < warn_v
    3. Separation: horizontal < nmac_h OR vertical < nmac_v

    Also supports:
    - Predictive conflict detection via ``predict_conflict()``
    - Multi-aircraft chain conflict detection via ``detect_chain_conflict()``
    """

    def __init__(self, config: dict[str, Any]) -> None:
        comp = config.get("components", {}).get("conflict", {})
        self._nmac_penalty: float = comp.get("nmac_penalty", -1.0)
        self._warning_penalty: float = comp.get("warning_penalty", 0.0)
        self._separation_penalty: float = comp.get("separation_penalty", 0.0)
        thresholds = comp.get("thresholds", {})
        self._nmac_h: float = thresholds.get("nmac_horizontal_nm", 5)
        self._nmac_v: float = thresholds.get("nmac_vertical_ft", 1000)
        self._warn_h: float = thresholds.get("warning_horizontal_nm", 10)
        self._warn_v: float = thresholds.get("warning_vertical_ft", 2000)
        self._routes: dict[str, Any] = {}  # agent_id → Route

    def set_routes(self, routes: dict[str, Any]) -> None:
        """Set route data for route-aware conflict detection.

        Args:
            routes: Mapping of agent_id to Route objects.
        """
        self._routes = routes

    def compute(
        self,
        agent_id: str,
        prev_state: AircraftState,
        action: DiscreteAction,
        curr_state: AircraftState,
        all_states: dict[str, AircraftState],
        step_count: int = 0,
    ) -> float:
        """Compute conflict penalty for the agent.

        Returns the most severe penalty among all conflicting aircraft.
        """
        worst = 0.0

        for other_id, other in all_states.items():
            if other_id == agent_id:
                continue

            h_dist = haversine_distance(
                curr_state.lat, curr_state.lon,
                other.lat, other.lon,
            )
            v_dist = abs(curr_state.alt - other.alt)

            if h_dist < self._nmac_h and v_dist < self._nmac_v:
                return self._nmac_penalty

            if h_dist < self._warn_h and v_dist < self._warn_v:
                worst = min(worst, self._warning_penalty)
            elif h_dist < self._nmac_h:
                worst = min(worst, self._separation_penalty)

        return worst

    def predict_conflict(
        self,
        own: AircraftState,
        other: AircraftState,
        lookahead_s: float = 60.0,
        conflict_distance_nm: float | None = None,
    ) -> bool:
        """Predict whether two aircraft will come into conflict within a time window.

        Projects both aircraft forward using their current heading and speed,
        sampling at 1-second intervals, and checks if horizontal distance
        drops below *conflict_distance_nm* (default: ``nmac_horizontal_nm``).

        Args:
            own: Ownship state.
            other: Other aircraft state.
            lookahead_s: Prediction horizon in seconds.
            conflict_distance_nm: Horizontal conflict threshold in NM.
                Defaults to the configured NMAC horizontal threshold.

        Returns:
            True if a conflict is predicted within the lookahead window.
        """
        if conflict_distance_nm is None:
            conflict_distance_nm = self._nmac_h

        dt = 1.0  # sample every 1 second
        steps = int(lookahead_s / dt)

        # Check current distance first
        cur_dist = haversine_distance(own.lat, own.lon, other.lat, other.lon)
        if cur_dist < conflict_distance_nm:
            return True

        # Project forward
        own_lat, own_lon = own.lat, own.lon
        oth_lat, oth_lon = other.lat, other.lon

        for _ in range(steps):
            own_lat, own_lon = project_position(
                own_lat, own_lon, own.hdg, own.tas, dt,
            )
            oth_lat, oth_lon = project_position(
                oth_lat, oth_lon, other.hdg, other.tas, dt,
            )
            dist = haversine_distance(own_lat, own_lon, oth_lat, oth_lon)
            if dist < conflict_distance_nm:
                return True

        return False

    def detect_chain_conflict(
        self,
        all_states: dict[str, AircraftState],
    ) -> list[list[str]]:
        """Detect groups of aircraft forming conflict chains.

        Two aircraft are linked if their horizontal distance < warning threshold
        AND vertical distance < warning threshold.  A chain is a connected
        component in this conflict graph.

        Args:
            all_states: All aircraft states keyed by agent ID.

        Returns:
            List of chains, each chain being a list of agent IDs.
        """
        ids = list(all_states.keys())
        n = len(ids)

        # Build adjacency: link aircraft that are within warning thresholds
        adj: dict[str, set[str]] = {aid: set() for aid in ids}
        for i in range(n):
            for j in range(i + 1, n):
                a, b = ids[i], ids[j]
                sa, sb = all_states[a], all_states[b]
                h_dist = haversine_distance(sa.lat, sa.lon, sb.lat, sb.lon)
                v_dist = abs(sa.alt - sb.alt)
                if h_dist < self._warn_h and v_dist < self._warn_v:
                    adj[a].add(b)
                    adj[b].add(a)

        # Find connected components via BFS
        visited: set[str] = set()
        chains: list[list[str]] = []
        for aid in ids:
            if aid in visited:
                continue
            # BFS
            queue = [aid]
            component: list[str] = []
            while queue:
                cur = queue.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                component.append(cur)
                for neighbor in adj[cur]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            if len(component) > 1:
                chains.append(sorted(component))

        return chains

    def reset(self) -> None:
        pass

    def get_conflict_status(
        self,
        own: AircraftState,
        others: list[AircraftState],
    ) -> str:
        """Get conflict status string for a pair of aircraft.

        Args:
            own: Ownship state.
            others: List of other aircraft states.

        Returns:
            "nmac", "warning", or "safe"
        """
        for other in others:
            h_dist = haversine_distance(own.lat, own.lon, other.lat, other.lon)
            v_dist = abs(own.alt - other.alt)
            if h_dist < self._nmac_h and v_dist < self._nmac_v:
                return "nmac"
            if h_dist < self._warn_h and v_dist < self._warn_v:
                return "warning"
        return "safe"

    def get_thresholds(self) -> dict[str, float]:
        """Get conflict thresholds.

        Returns:
            Dictionary with nmac_h, nmac_v, warn_h, warn_v
        """
        return {
            "nmac_h": self._nmac_h,
            "nmac_v": self._nmac_v,
            "warn_h": self._warn_h,
            "warn_v": self._warn_v,
        }
