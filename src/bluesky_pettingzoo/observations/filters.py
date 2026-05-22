"""Perception range filter for partial observability."""

from __future__ import annotations

from typing import Any

from bluesky_pettingzoo.utils.geometry import haversine_distance, bearing
from bluesky_pettingzoo.utils.types import AircraftState


class PerceptionFilter:
    """Filters aircraft based on perception range constraints.

    Applies horizontal radius, vertical range, and max observable limits.
    Results are sorted by distance (ascending).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        obs = config.get("observation", {})
        self._radius_nm: float = obs.get("perception_radius_nm", 20)
        self._alt_diff_ft: float = obs.get("perception_alt_diff_ft", 3000)
        self._max_observable: int = obs.get("max_observable_aircraft", 10)

    def filter(
        self,
        own_state: AircraftState,
        others: list[AircraftState],
    ) -> list[dict[str, Any]]:
        """Filter aircraft within perception range.

        Args:
            own_state: Ownship aircraft state.
            others: List of other aircraft states.

        Returns:
            List of dicts with keys: state, distance_nm, bearing_deg.
            Sorted by distance ascending, truncated to max observable.
        """
        results: list[dict[str, Any]] = []

        for other in others:
            dist = haversine_distance(
                own_state["lat"], own_state["lon"],
                other["lat"], other["lon"],
            )
            # Tolerance for floating-point and coordinate precision
            if dist > self._radius_nm + 0.05:
                continue

            alt_diff = abs(own_state["alt"] - other["alt"])
            if alt_diff > self._alt_diff_ft:
                continue

            bear = bearing(
                own_state["lat"], own_state["lon"],
                other["lat"], other["lon"],
            )
            results.append({
                "state": other,
                "distance_nm": dist,
                "bearing_deg": bear,
            })

        results.sort(key=lambda r: r["distance_nm"])
        return results[: self._max_observable]
