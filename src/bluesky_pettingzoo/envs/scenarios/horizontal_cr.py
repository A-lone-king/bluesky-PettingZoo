"""Horizontal conflict resolution scenario.

Multiple aircraft cruise at the same altitude, use heading maneuvers
to avoid conflicts, and terminate upon reaching waypoints.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.geometry import point_at_distance
from bluesky_pettingzoo.utils.types import AircraftState, ConflictConfig, SpawnConfig

# Constants matching bluesky-gym reference
WAYPOINT_DISTANCE_MIN_NM = 100
WAYPOINT_DISTANCE_MAX_NM = 150
CRUISE_ALT_FT = 35000.0

# Multi-altitude layer constants
ALTITUDE_LAYER_MIN_FT = 29000.0
ALTITUDE_LAYER_MAX_FT = 41000.0
ALTITUDE_LAYER_SEPARATION_FT = 4000.0  # Minimum separation between layers


class HorizontalCRScenario(BaseScenario):
    """Horizontal conflict resolution scenario.

    Multiple aircraft cruise at the same altitude, use heading maneuvers
    to avoid conflicts, and terminate upon reaching waypoints.

    Args:
        num_aircraft: Number of aircraft to spawn.
        num_aircraft_range: Optional (min, max) for procedural generation.
        seed: Optional seed for reproducibility (used if rng not provided).
    """

    def __init__(
        self,
        num_aircraft: int = 5,
        num_aircraft_range: tuple[int, int] | None = None,
        seed: int | None = None,
        num_altitude_layers: int = 1,
        waypoint_distance_range: tuple[float, float] | None = None,
    ) -> None:
        self._num_aircraft = num_aircraft
        self._num_aircraft_range = num_aircraft_range
        self._seed = seed
        self._num_altitude_layers = num_altitude_layers
        self._waypoint_distance_range = waypoint_distance_range
        self._agents: list[str] = []
        self._waypoints: dict[str, dict[str, float]] = {}
        self._bounds: dict[str, float] = {}

    @property
    def action_dimensions(self) -> list[int]:
        """Return which action indices are valid (0=heading, 1=altitude, 2=speed)."""
        return [0]  # heading only

    @property
    def num_aircraft_range(self) -> tuple[int, int] | None:
        """Return dynamic aircraft count range if configured."""
        return self._num_aircraft_range

    def reset(self, rng: np.random.RandomState) -> None:
        """Randomize aircraft count for procedural generation."""
        if self._num_aircraft_range is not None:
            self._num_aircraft = int(
                rng.randint(
                    self._num_aircraft_range[0],
                    self._num_aircraft_range[1] + 1,
                )
            )
        self._agents = []
        self._waypoints = {}

    def setup(
        self,
        rng: np.random.RandomState,
        airspace_bounds: dict[str, float],
    ) -> list[str]:
        """Initialize scenario: generate agent IDs and assign waypoints.

        Aircraft are placed at random positions; waypoints are placed
        100-150 NM away in alternating east/west directions to create
        head-on conflict opportunities. When num_altitude_layers > 1,
        aircraft are distributed across multiple altitude layers.
        """
        self._bounds = airspace_bounds

        # Use configurable waypoint distance range if provided
        wp_min = WAYPOINT_DISTANCE_MIN_NM
        wp_max = WAYPOINT_DISTANCE_MAX_NM
        if self._waypoint_distance_range is not None:
            wp_min = self._waypoint_distance_range[0]
            wp_max = self._waypoint_distance_range[1]

        self._agents = [f"AC{i:03d}" for i in range(self._num_aircraft)]
        self._waypoints = {}

        # Generate altitude layers
        if self._num_altitude_layers == 1:
            altitude_layers = [CRUISE_ALT_FT]
        else:
            # Evenly distribute layers across altitude range
            altitude_layers = np.linspace(
                ALTITUDE_LAYER_MIN_FT,
                ALTITUDE_LAYER_MAX_FT,
                self._num_altitude_layers,
            ).tolist()

        # Assign aircraft to layers (distribute evenly)
        layer_assignments = []
        for i in range(self._num_aircraft):
            layer_idx = i % len(altitude_layers)
            layer_assignments.append(altitude_layers[layer_idx])

        for i, acid in enumerate(self._agents):
            # Place aircraft at random position within airspace
            lat_min = airspace_bounds["lat_min"] + 0.05
            lat_max = airspace_bounds["lat_max"] - 0.05
            ac_lat = rng.uniform(lat_min, lat_max)

            lon_min = airspace_bounds["lon_min"] + 0.05
            lon_max = airspace_bounds["lon_max"] - 0.05
            ac_lon = rng.uniform(lon_min, lon_max)

            # Alternate waypoint direction: even → east, odd → west
            # This creates head-on conflict opportunities
            if i % 2 == 0:
                bearing_deg = rng.uniform(60, 120)  # roughly eastward
            else:
                bearing_deg = rng.uniform(240, 300)  # roughly westward

            dist_nm = rng.uniform(wp_min, wp_max)
            wp_lat, wp_lon = point_at_distance(ac_lat, ac_lon, dist_nm, bearing_deg)

            self._waypoints[acid] = {
                "lat": wp_lat,
                "lon": wp_lon,
                "alt": layer_assignments[i],
                "hdg": bearing_deg,
            }

        return list(self._agents)

    def get_spawn_config(self) -> SpawnConfig:
        """All aircraft cruise at the same altitude (horizontal CR)."""
        return SpawnConfig(
            altitude_range=(CRUISE_ALT_FT, CRUISE_ALT_FT),
            speed_range=(400, 500),
            heading_range=(0, 360),
        )

    def get_conflict_config(self) -> ConflictConfig:
        """Standard conflict thresholds for horizontal CR."""
        return ConflictConfig(
            nmac_horizontal_nm=5.0,
            nmac_vertical_ft=1000.0,
            warning_horizontal_nm=10.0,
            warning_vertical_ft=2000.0,
        )

    def get_waypoint(self, agent_id: str) -> dict[str, float]:
        """Return the assigned waypoint for an agent."""
        return self._waypoints[agent_id]

    def get_priority(self, agent_id: str, state: AircraftState) -> float:
        """Higher altitude → higher priority (normalized to [-1, 1])."""
        alt_min, alt_max = 29000.0, 37000.0
        alt_mid = (alt_min + alt_max) / 2
        alt_half = (alt_max - alt_min) / 2
        return max(-1.0, min(1.0, (state.alt - alt_mid) / alt_half))

    def create_intruders(self, wrapper: Any, rng: np.random.RandomState | None = None) -> list[str]:
        """Create intruder aircraft using creconfs for head-on conflicts.

        Args:
            wrapper: BlueSkyWrapper instance.
            rng: Random number generator (optional).

        Returns:
            List of created aircraft IDs (ownship + intruders).
        """
        mid_lat = (self._bounds["lat_min"] + self._bounds["lat_max"]) / 2
        mid_lon = (self._bounds["lon_min"] + self._bounds["lon_max"]) / 2

        dpsi = 150.0 + (rng.uniform(-20, 20) if rng is not None else 0)
        dcpa = 3.0 + (rng.uniform(-1, 1) if rng is not None else 0)

        return wrapper.create_conflict_aircraft(  # type: ignore[no-any-return]
            ownship_lat=mid_lat,
            ownship_lon=mid_lon,
            ownship_alt=CRUISE_ALT_FT,
            ownship_hdg=90.0,
            ownship_spd=450.0,
            count=self._num_aircraft - 1,
            dpsi=dpsi,
            dcpa=max(dcpa, 1.0),
            prefix="CR",
        )
