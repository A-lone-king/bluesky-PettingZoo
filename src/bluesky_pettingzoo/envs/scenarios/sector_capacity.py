"""Sector capacity management scenario.

Multiple aircraft navigate through sectors with capacity constraints.
Aircraft must reach their waypoints without exceeding sector capacity
limits. Combines conflict resolution with traffic management.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.geometry import assign_sector
from bluesky_pettingzoo.utils.types import AircraftState, ConflictConfig, SpawnConfig

CRUISE_ALT_FT = 35000.0
SPEED_MIN_KT = 400.0
SPEED_MAX_KT = 500.0


class SectorCapacityScenario(BaseScenario):
    """Sector capacity management scenario.

    Aircraft navigate through a set of sectors with capacity constraints.
    Each sector has a maximum number of aircraft allowed simultaneously.
    The goal is to reach waypoints while respecting capacity limits.

    Args:
        num_aircraft: Number of aircraft to spawn.
        num_sectors: Number of sectors (default 2).
        sector_capacity: Maximum aircraft per sector (default 4).
        seed: Optional seed for reproducibility.
    """

    def __init__(
        self,
        num_aircraft: int = 6,
        num_aircraft_range: tuple[int, int] | None = None,
        num_sectors: int = 2,
        sector_capacity: int = 4,
        seed: int | None = None,
    ) -> None:
        self._num_aircraft = num_aircraft
        self._num_aircraft_range = num_aircraft_range
        self._num_sectors = num_sectors
        self._sector_capacity = sector_capacity
        self._seed = seed
        self._agents: list[str] = []
        self._waypoints: dict[str, dict[str, float]] = {}
        self._sectors: list[dict[str, Any]] = []
        self._initial_positions: dict[str, tuple[float, float]] | None = None

    @property
    def action_dimensions(self) -> list[int]:
        """Heading + speed adjustments."""
        return [0, 2]

    @property
    def num_aircraft_range(self) -> tuple[int, int] | None:
        """Return dynamic aircraft count range if configured."""
        return self._num_aircraft_range

    def reset(self, rng: np.random.RandomState) -> None:
        """Randomize aircraft count and clear state for procedural generation."""
        if self._num_aircraft_range is not None:
            self._num_aircraft = int(
                rng.randint(
                    self._num_aircraft_range[0],
                    self._num_aircraft_range[1] + 1,
                )
            )
        self._agents = []
        self._waypoints = {}
        self._sectors = []
        self._initial_positions = None

    def get_sectors(self) -> list[dict[str, Any]]:
        """Return sector definitions with capacity."""
        return self._sectors

    def get_initial_positions(self) -> dict[str, tuple[float, float]] | None:
        """Return initial lat/lon positions for all agents, or None if unset."""
        return self._initial_positions

    def setup(
        self,
        rng: np.random.RandomState,
        airspace_bounds: dict[str, float],
    ) -> list[str]:
        """Initialize sectors and place aircraft in the first sector."""
        self._bounds = airspace_bounds
        self._agents = [f"AC{i:03d}" for i in range(self._num_aircraft)]
        self._waypoints = {}

        lon_span = airspace_bounds["lon_max"] - airspace_bounds["lon_min"]

        # Generate sectors as adjacent rectangles along longitude
        sector_width = lon_span / self._num_sectors
        self._sectors = []
        for i in range(self._num_sectors):
            lon_min = airspace_bounds["lon_min"] + i * sector_width
            lon_max = lon_min + sector_width
            self._sectors.append(
                {
                    "id": f"sector_{i}",
                    "bounds": [
                        [airspace_bounds["lat_min"], lon_min],
                        [airspace_bounds["lat_max"], lon_max],
                    ],
                    "capacity": self._sector_capacity,
                }
            )

        # Place aircraft in the first sector
        first = self._sectors[0]
        (s_lat_min, s_lon_min), (s_lat_max, s_lon_max) = first["bounds"]
        self._initial_positions = {}
        for acid in self._agents:
            ac_lat = rng.uniform(s_lat_min + 0.05, s_lat_max - 0.05)
            ac_lon = rng.uniform(s_lon_min + 0.05, s_lon_max - 0.05)
            self._initial_positions[acid] = (ac_lat, ac_lon)

            # Assign waypoint in the last sector
            last = self._sectors[-1]
            (_, w_lon_min), (_, w_lon_max) = last["bounds"]
            wp_lat = rng.uniform(airspace_bounds["lat_min"] + 0.1, airspace_bounds["lat_max"] - 0.1)
            wp_lon = rng.uniform(w_lon_min + 0.1, w_lon_max - 0.1)
            self._waypoints[acid] = {
                "lat": wp_lat,
                "lon": wp_lon,
                "alt": CRUISE_ALT_FT,
                "hdg": rng.uniform(0, 360),
            }

        return list(self._agents)

    def get_spawn_config(self) -> SpawnConfig:
        """Return aircraft spawn configuration (altitude, speed, heading ranges)."""
        return SpawnConfig(
            altitude_range=(CRUISE_ALT_FT, CRUISE_ALT_FT),
            speed_range=(SPEED_MIN_KT, SPEED_MAX_KT),
            heading_range=(0, 360),
        )

    def get_conflict_config(self) -> ConflictConfig:
        """Return conflict detection thresholds (NMAC and warning distances)."""
        return ConflictConfig(
            nmac_horizontal_nm=5.0,
            nmac_vertical_ft=1000.0,
            warning_horizontal_nm=10.0,
            warning_vertical_ft=2000.0,
        )

    def should_truncate(
        self,
        agent_id: str,
        state: AircraftState,
        airspace_bounds: dict[str, float],
    ) -> bool:
        """Truncate aircraft that leave all sectors."""
        sector = assign_sector(state.lat, state.lon, self._sectors)
        return sector is None

    def get_waypoint(self, agent_id: str) -> dict[str, float]:
        """Return the goal waypoint for the given agent.

        Args:
            agent_id: The agent identifier.

        Returns:
            Dictionary with lat, lon, alt, hdg keys.
        """
        return self._waypoints[agent_id]
