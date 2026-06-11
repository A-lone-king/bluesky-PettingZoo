"""Vertical conflict resolution scenario.

Multiple aircraft at similar horizontal positions but different altitudes,
use vertical speed maneuvers to avoid conflicts.
Conflict requires BOTH horizontal < 5 NM AND vertical < 1000 ft.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.geometry import point_at_distance
from bluesky_pettingzoo.utils.types import AircraftState, ConflictConfig, SpawnConfig

# Constants matching bluesky-gym reference
ALT_MIN_FT = 20000.0
ALT_MAX_FT = 40000.0
ALT_SEPARATION_FT = 3000.0  # Minimum altitude separation between aircraft

# Approach profile constants
GLIDE_SLOPE_DEG = 3.0  # Standard 3° glide slope
FINAL_APPROACH_ALT_FT = 3000.0  # Final approach altitude (runway threshold)
APPROACH_SPEED_INITIAL_KT = 250.0  # Initial approach speed
APPROACH_SPEED_FINAL_KT = 140.0  # Final approach speed (Vref)
TARGET_VS_FT_MIN = -700.0  # Target vertical speed for 3° glide slope at 140 kts


class VerticalCRScenario(BaseScenario):
    """Vertical conflict resolution scenario.

    Aircraft are placed at similar horizontal positions but different altitudes.
    Conflict detection requires both horizontal (< 5 NM) and vertical (< 1000 ft)
    thresholds to be violated simultaneously.

    Args:
        num_aircraft: Number of aircraft to spawn.
        seed: Optional seed for reproducibility.
    """

    def __init__(
        self,
        num_aircraft: int = 5,
        num_aircraft_range: tuple[int, int] | None = None,
        seed: int | None = None,
        use_approach_profile: bool = False,
    ) -> None:
        self._num_aircraft = num_aircraft
        self._num_aircraft_range = num_aircraft_range
        self._seed = seed
        self._use_approach_profile = use_approach_profile
        self._agents: list[str] = []
        self._waypoints: dict[str, dict[str, float]] = {}
        self._bounds: dict[str, float] = {}
        self._approach_profiles: dict[str, dict[str, float]] = {}

    @property
    def action_dimensions(self) -> list[int]:
        """Return which action indices are valid (0=heading, 1=altitude, 2=speed)."""
        return [1]  # altitude/vertical speed only

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
        """Initialize scenario: aircraft at similar horizontal positions, different altitudes.

        Aircraft are clustered horizontally to create vertical conflict opportunities.
        Each aircraft gets a target altitude that differs from its current altitude.
        When use_approach_profile is True, aircraft follow a 3° glide slope approach.
        """
        self._bounds = airspace_bounds
        self._agents = [f"AC{i:03d}" for i in range(self._num_aircraft)]
        self._waypoints = {}
        self._approach_profiles = {}

        mid_lat = (airspace_bounds["lat_min"] + airspace_bounds["lat_max"]) / 2
        mid_lon = (airspace_bounds["lon_min"] + airspace_bounds["lon_max"]) / 2

        if self._use_approach_profile:
            # Create approach profiles for each aircraft
            # Aircraft start at different distances from runway, converge on same path
            for i, acid in enumerate(self._agents):
                # Random distance from runway (5-15 NM)
                distance_nm = rng.uniform(5.0, 15.0)
                # Random bearing from runway
                bearing_deg = rng.uniform(0, 360)

                # Calculate starting position (far from runway)
                start_lat, start_lon = point_at_distance(
                    mid_lat, mid_lon, distance_nm, bearing_deg
                )

                # Speed decreases as aircraft approaches runway
                initial_speed = APPROACH_SPEED_INITIAL_KT
                final_speed = APPROACH_SPEED_FINAL_KT

                self._approach_profiles[acid] = {
                    "glide_slope_deg": GLIDE_SLOPE_DEG,
                    "initial_speed_kt": initial_speed,
                    "final_speed_kt": final_speed,
                    "final_alt_ft": FINAL_APPROACH_ALT_FT,
                    "target_vs_ft_min": TARGET_VS_FT_MIN,
                    "distance_nm": distance_nm,
                }

                # Waypoint is runway threshold (final position)
                self._waypoints[acid] = {
                    "lat": mid_lat,
                    "lon": mid_lon,
                    "alt": FINAL_APPROACH_ALT_FT,
                    "hdg": (bearing_deg + 180) % 360,  # Heading towards runway
                }
        else:
            # Original behavior: aircraft at different altitudes
            altitudes = np.linspace(ALT_MIN_FT, ALT_MAX_FT, self._num_aircraft)
            rng.shuffle(altitudes)

            for i, acid in enumerate(self._agents):
                # Place aircraft in a small horizontal cluster (within ~2 NM)
                # This creates the horizontal proximity needed for vertical CR
                ac_lat = mid_lat + rng.uniform(-0.02, 0.02)
                ac_lon = mid_lon + rng.uniform(-0.02, 0.02)

                # Target altitude: swap with another aircraft's altitude to create
                # vertical maneuvering need
                target_alt = float(altitudes[(i + 1) % self._num_aircraft])

                self._waypoints[acid] = {
                    "lat": ac_lat + rng.uniform(-0.05, 0.05),
                    "lon": ac_lon + rng.uniform(-0.05, 0.05),
                    "alt": target_alt,
                    "hdg": rng.uniform(0, 360),
                }

        return list(self._agents)

    def should_truncate(
        self,
        agent_id: str,
        state: AircraftState,
        airspace_bounds: dict[str, float],
    ) -> bool:
        """Check if agent should be truncated.

        For approach profile mode: truncate when within 500 ft of final altitude.
        Otherwise: use default rectangular bounds check.
        """
        if self._use_approach_profile and agent_id in self._approach_profiles:
            profile = self._approach_profiles[agent_id]
            final_alt = profile["final_alt_ft"]
            # Truncate when within 500 ft of final altitude
            return abs(state.alt - final_alt) < 500.0
        # Default: truncate when leaving airspace bounds
        return (
            state.lat < airspace_bounds["lat_min"]
            or state.lat > airspace_bounds["lat_max"]
            or state.lon < airspace_bounds["lon_min"]
            or state.lon > airspace_bounds["lon_max"]
        )

    def get_spawn_config(self) -> SpawnConfig:
        """Aircraft spawn at varying altitudes (vertical CR)."""
        return SpawnConfig(
            altitude_range=(ALT_MIN_FT, ALT_MAX_FT),
            speed_range=(400, 500),
            heading_range=(0, 360),
        )

    def get_conflict_config(self) -> ConflictConfig:
        """Standard conflict thresholds for vertical CR."""
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
        """Faster speed → higher priority (normalized to [-1, 1])."""
        speed_min, speed_max = 400.0, 500.0
        speed_mid = (speed_min + speed_max) / 2
        speed_half = (speed_max - speed_min) / 2
        return max(-1.0, min(1.0, (state.tas - speed_mid) / speed_half))

    def create_intruders(self, wrapper: Any, rng: np.random.RandomState | None = None) -> list[str]:
        """Create intruder aircraft using creconfs with vertical offset.

        Args:
            wrapper: BlueSkyWrapper instance.
            rng: Random number generator (optional).

        Returns:
            List of created aircraft IDs (ownship + intruders).
        """
        mid_lat = (self._bounds["lat_min"] + self._bounds["lat_max"]) / 2
        mid_lon = (self._bounds["lon_min"] + self._bounds["lon_max"]) / 2
        base_alt = (ALT_MIN_FT + ALT_MAX_FT) / 2

        return wrapper.create_conflict_aircraft(  # type: ignore[no-any-return]
            ownship_lat=mid_lat,
            ownship_lon=mid_lon,
            ownship_alt=base_alt,
            ownship_hdg=rng.uniform(0, 360) if rng is not None else 0.0,
            ownship_spd=450.0,
            count=self._num_aircraft - 1,
            dpsi=rng.uniform(10, 45) if rng is not None else 30.0,
            dcpa=3.0,
            dH=rng.uniform(-3000, 3000) if rng is not None else 2000.0,
            prefix="CR",
        )
