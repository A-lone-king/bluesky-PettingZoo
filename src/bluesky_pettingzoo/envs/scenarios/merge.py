"""Approach merge scenario.

1 controllable aircraft + 19 background traffic on approach.
Background traffic follows preset routes (uncontrollable).
The controllable aircraft must find a safe gap to merge.
Conflict distance: 4 NM (stricter than cruise).
Observable neighbors limited to 5.
"""

from __future__ import annotations

import numpy as np

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.geometry import point_at_distance
from bluesky_pettingzoo.utils.types import AircraftState, ConflictConfig, SpawnConfig

# Approach constants
APPROACH_ALT_MIN_FT = 3000.0
APPROACH_ALT_MAX_FT = 8000.0
SPEED_MIN_KT = 200.0
SPEED_MAX_KT = 280.0
NMAC_HORIZONTAL_NM = 4.0
WARNING_HORIZONTAL_NM = 8.0


class MergeScenario(BaseScenario):
    """Approach merge scenario with background traffic.

    Generates 1 controllable aircraft and (num_aircraft - 1) background
    aircraft on approach paths.  Background traffic follows preset routes
    and is not controllable by the agent.

    Args:
        num_aircraft: Total aircraft count (default 20).
        seed: Optional seed for reproducibility.
    """

    def __init__(self, num_aircraft: int = 20, seed: int | None = None) -> None:
        self._num_aircraft = num_aircraft
        self._seed = seed
        self._agents: list[str] = []
        self._controllable: list[str] = []
        self._background: list[str] = []
        self._waypoints: dict[str, dict[str, float]] = {}
        self._bounds: dict[str, float] = {}

    @property
    def action_dimensions(self) -> list[int]:
        """All action axes available (heading, altitude, speed)."""
        return [0, 1, 2]

    def get_controllable_agents(self) -> list[str]:
        """Return the list of controllable agent IDs."""
        return list(self._controllable)

    def get_background_agents(self) -> list[str]:
        """Return the list of background (uncontrollable) agent IDs."""
        return list(self._background)

    def setup(
        self,
        rng: np.random.RandomState,
        airspace_bounds: dict[str, float],
    ) -> list[str]:
        """Initialize scenario: 1 controllable + N-1 background aircraft.

        All aircraft are on approach paths toward a common FAF point.
        Background aircraft are spaced along approach corridors.
        """
        self._bounds = airspace_bounds
        self._agents = [f"AC{i:03d}" for i in range(self._num_aircraft)]
        self._controllable = [self._agents[0]]
        self._background = self._agents[1:]
        self._waypoints = {}

        center_lat = (airspace_bounds["lat_min"] + airspace_bounds["lat_max"]) / 2
        center_lon = (airspace_bounds["lon_min"] + airspace_bounds["lon_max"]) / 2

        # FAF (Final Approach Fix) at center
        faf_lat = center_lat
        faf_lon = center_lon

        # Controllable aircraft: start from south, heading north toward FAF
        ctrl = self._controllable[0]
        self._waypoints[ctrl] = {
            "lat": faf_lat,
            "lon": faf_lon,
            "alt": rng.uniform(APPROACH_ALT_MIN_FT, APPROACH_ALT_MAX_FT),
            "hdg": 0.0,  # heading north
        }

        # Background aircraft: distributed along approach corridors
        for i, acid in enumerate(self._background):
            # Spread along different approach bearings (0-360 degrees)
            bearing = (i * 360.0 / len(self._background)) % 360.0
            dist_nm = rng.uniform(15, 30)  # 15-30 NM from FAF
            ac_lat, ac_lon = point_at_distance(faf_lat, faf_lon, dist_nm, bearing)
            hdg_to_faf = (bearing + 180) % 360  # heading toward FAF

            self._waypoints[acid] = {
                "lat": faf_lat,
                "lon": faf_lon,
                "alt": rng.uniform(APPROACH_ALT_MIN_FT, APPROACH_ALT_MAX_FT),
                "hdg": hdg_to_faf,
            }

        return list(self._agents)

    def get_spawn_config(self) -> SpawnConfig:
        """Approach altitude and speed."""
        return SpawnConfig(
            altitude_range=(APPROACH_ALT_MIN_FT, APPROACH_ALT_MAX_FT),
            speed_range=(SPEED_MIN_KT, SPEED_MAX_KT),
            heading_range=(0, 360),
        )

    def get_conflict_config(self) -> ConflictConfig:
        """Stricter conflict thresholds for approach (4 NM NMAC)."""
        return ConflictConfig(
            nmac_horizontal_nm=NMAC_HORIZONTAL_NM,
            nmac_vertical_ft=500.0,
            warning_horizontal_nm=WARNING_HORIZONTAL_NM,
            warning_vertical_ft=1000.0,
        )

    def get_waypoint(self, agent_id: str) -> dict[str, float]:
        """Return the assigned FAF waypoint for an agent."""
        return self._waypoints[agent_id]
