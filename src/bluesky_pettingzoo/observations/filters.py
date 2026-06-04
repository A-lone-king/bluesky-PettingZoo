"""Perception range filter for partial observability."""

from __future__ import annotations

from typing import Any

import numpy as np

from bluesky_pettingzoo.utils.geometry import (
    bearing,
    haversine_distance,
    haversine_distance_matrix,
)
from bluesky_pettingzoo.utils.types import AircraftState


class PerceptionFilter:
    """Filters aircraft based on perception range constraints.

    Applies horizontal radius, vertical range, and max observable limits.
    Uses vectorized distance matrix for efficient batch computation.
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

        Uses vectorized distance matrix when multiple others exist,
        falls back to scalar computation for single aircraft.

        Args:
            own_state: Ownship aircraft state.
            others: List of other aircraft states.

        Returns:
            List of dicts with keys: state, distance_nm, bearing_deg.
            Sorted by distance ascending, truncated to max observable.
        """
        if not others:
            return []

        if len(others) > 1:
            return self._filter_vectorized(own_state, others)

        return self._filter_scalar(own_state, others[0])

    def _filter_vectorized(
        self,
        own_state: AircraftState,
        others: list[AircraftState],
    ) -> list[dict[str, Any]]:
        """Vectorized filtering using distance matrix."""
        n = len(others)

        lats = np.array([own_state["lat"]] + [o["lat"] for o in others])
        lons = np.array([own_state["lon"]] + [o["lon"] for o in others])

        dist_matrix = haversine_distance_matrix(lats, lons)
        distances = dist_matrix[0, 1:]

        own_alt = own_state["alt"]
        alt_diffs = np.array([abs(own_alt - o["alt"]) for o in others])

        radius_with_tol = self._radius_nm + 0.05
        mask = (distances <= radius_with_tol) & (alt_diffs <= self._alt_diff_ft)

        results: list[dict[str, Any]] = []
        for i in range(n):
            if not mask[i]:
                continue
            bear = bearing(
                own_state["lat"],
                own_state["lon"],
                others[i]["lat"],
                others[i]["lon"],
            )
            results.append(
                {
                    "state": others[i],
                    "distance_nm": float(distances[i]),
                    "bearing_deg": bear,
                }
            )

        results.sort(key=lambda r: r["distance_nm"])
        return results[: self._max_observable]

    def _filter_scalar(
        self,
        own_state: AircraftState,
        other: AircraftState,
    ) -> list[dict[str, Any]]:
        """Scalar filtering for single aircraft."""
        dist = haversine_distance(
            own_state["lat"],
            own_state["lon"],
            other["lat"],
            other["lon"],
        )
        if dist > self._radius_nm + 0.05:
            return []

        alt_diff = abs(own_state["alt"] - other["alt"])
        if alt_diff > self._alt_diff_ft:
            return []

        bear = bearing(
            own_state["lat"],
            own_state["lon"],
            other["lat"],
            other["lon"],
        )
        return [
            {
                "state": other,
                "distance_nm": dist,
                "bearing_deg": bear,
            }
        ]
