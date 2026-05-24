"""Vertical conflict resolution scenario.

Multiple aircraft at similar horizontal positions but different altitudes,
use vertical speed maneuvers to avoid conflicts.
Conflict requires BOTH horizontal < 5 NM AND vertical < 1000 ft.
"""

from __future__ import annotations

import numpy as np

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.types import AircraftState, ConflictConfig, SpawnConfig

# Constants matching bluesky-gym reference
ALT_MIN_FT = 20000.0
ALT_MAX_FT = 40000.0
ALT_SEPARATION_FT = 3000.0  # Minimum altitude separation between aircraft


class VerticalCRScenario(BaseScenario):
    """Vertical conflict resolution scenario.

    Aircraft are placed at similar horizontal positions but different altitudes.
    Conflict detection requires both horizontal (< 5 NM) and vertical (< 1000 ft)
    thresholds to be violated simultaneously.

    Args:
        num_aircraft: Number of aircraft to spawn.
        seed: Optional seed for reproducibility.
    """

    def __init__(self, num_aircraft: int = 5, seed: int | None = None) -> None:
        self._num_aircraft = num_aircraft
        self._seed = seed
        self._agents: list[str] = []
        self._waypoints: dict[str, dict[str, float]] = {}
        self._bounds: dict[str, float] = {}

    @property
    def action_dimensions(self) -> list[int]:
        """Return which action indices are valid (0=heading, 1=altitude, 2=speed)."""
        return [1]  # altitude/vertical speed only

    def setup(
        self,
        rng: np.random.RandomState,
        airspace_bounds: dict[str, float],
    ) -> list[str]:
        """Initialize scenario: aircraft at similar horizontal positions, different altitudes.

        Aircraft are clustered horizontally to create vertical conflict opportunities.
        Each aircraft gets a target altitude that differs from its current altitude.
        """
        self._bounds = airspace_bounds
        self._agents = [f"AC{i:03d}" for i in range(self._num_aircraft)]
        self._waypoints = {}

        mid_lat = (airspace_bounds["lat_min"] + airspace_bounds["lat_max"]) / 2
        mid_lon = (airspace_bounds["lon_min"] + airspace_bounds["lon_max"]) / 2

        # Generate evenly spaced altitudes across the range
        altitudes = np.linspace(ALT_MIN_FT, ALT_MAX_FT, self._num_aircraft)
        rng.shuffle(altitudes)

        for i, acid in enumerate(self._agents):
            # Place aircraft in a small horizontal cluster (within ~2 NM)
            # This creates the horizontal proximity needed for vertical CR
            ac_lat = mid_lat + rng.uniform(-0.02, 0.02)
            ac_lon = mid_lon + rng.uniform(-0.02, 0.02)

            current_alt = float(altitudes[i])
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
        return max(-1.0, min(1.0, (state.tas - (speed_min + speed_max) / 2) / ((speed_max - speed_min) / 2)))
