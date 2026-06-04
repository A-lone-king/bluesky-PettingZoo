"""Descent approach scenario.

Multiple aircraft at cruising altitude must descend to target altitudes
before reaching the runway.  Vertical-speed-only control.
"""

from __future__ import annotations

import numpy as np

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.types import AircraftState, ConflictConfig, SpawnConfig

# Constants matching bluesky-gym reference
ALT_MIN_FT = 20000.0
ALT_MAX_FT = 40000.0
TARGET_ALT_MIN = 2000.0
TARGET_ALT_MAX = 6000.0
AC_SPD = 150.0  # knots, approach speed


class DescentScenario(BaseScenario):
    """Descent approach scenario.

    Aircraft are spawned at cruising altitude along an approach path toward
    a runway positioned at the center of the airspace.  Each aircraft has a
    target altitude it must reach before crossing the runway threshold.

    Crash detection: ``alt <= 0`` triggers truncation.

    Args:
        num_aircraft: Number of aircraft to spawn.
        seed: Optional seed for reproducibility.
    """

    def __init__(
        self,
        num_aircraft: int = 3,
        num_aircraft_range: tuple[int, int] | None = None,
        seed: int | None = None,
    ) -> None:
        self._num_aircraft = num_aircraft
        self._num_aircraft_range = num_aircraft_range
        self._seed = seed
        self._agents: list[str] = []
        self._waypoints: dict[str, dict[str, float]] = {}
        self._bounds: dict[str, float] = {}

    @property
    def control_mode(self) -> str:
        """Descent uses SINGLE_RL: only the first aircraft is agent-controlled."""
        return "SINGLE_RL"

    @property
    def ego_agent(self) -> str | None:
        """The controllable aircraft."""
        return self._agents[0] if self._agents else None

    @property
    def background_agents(self) -> list[str]:
        """Background traffic (uncontrollable)."""
        return list(self._agents[1:]) if len(self._agents) > 1 else []

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
            self._num_aircraft = int(rng.randint(
                self._num_aircraft_range[0],
                self._num_aircraft_range[1] + 1,
            ))
        self._agents = []
        self._waypoints = {}

    def setup(
        self,
        rng: np.random.RandomState,
        airspace_bounds: dict[str, float],
    ) -> list[str]:
        """Initialize scenario: aircraft at cruising altitude, runway at center.

        Each aircraft is placed at a random position within the airspace at
        cruising altitude and assigned a low target altitude (descent goal).
        """
        self._bounds = airspace_bounds
        self._agents = [f"AC{i:03d}" for i in range(self._num_aircraft)]
        self._waypoints = {}

        # Runway at center of airspace
        rwy_lat = (airspace_bounds["lat_min"] + airspace_bounds["lat_max"]) / 2
        rwy_lon = (airspace_bounds["lon_min"] + airspace_bounds["lon_max"]) / 2

        # Generate target altitudes
        target_alts = np.linspace(TARGET_ALT_MIN, TARGET_ALT_MAX, self._num_aircraft)
        rng.shuffle(target_alts)

        for i, acid in enumerate(self._agents):
            # Aircraft heading toward runway
            hdg_to_rwy = rng.uniform(0, 360)

            self._waypoints[acid] = {
                "lat": rwy_lat,
                "lon": rwy_lon,
                "alt": float(target_alts[i]),
                "hdg": hdg_to_rwy,
            }

        return list(self._agents)

    def get_spawn_config(self) -> SpawnConfig:
        """Aircraft spawn at cruising altitude with approach speed."""
        return SpawnConfig(
            altitude_range=(ALT_MIN_FT, ALT_MAX_FT),
            speed_range=(AC_SPD, AC_SPD + 50),
            heading_range=(0, 360),
        )

    def get_conflict_config(self) -> ConflictConfig:
        """Standard conflict thresholds."""
        return ConflictConfig(
            nmac_horizontal_nm=5.0,
            nmac_vertical_ft=1000.0,
            warning_horizontal_nm=10.0,
            warning_vertical_ft=2000.0,
        )

    def get_waypoint(self, agent_id: str) -> dict[str, float]:
        """Return the runway waypoint with target altitude."""
        return self._waypoints[agent_id]

    def should_truncate(
        self,
        agent_id: str,
        state: AircraftState,
        airspace_bounds: dict[str, float],
    ) -> bool:
        """Truncate on crash (alt <= 0) or leaving airspace bounds."""
        if state.alt <= 0:
            return True
        return (
            state.lat < airspace_bounds["lat_min"]
            or state.lat > airspace_bounds["lat_max"]
            or state.lon < airspace_bounds["lon_min"]
            or state.lon > airspace_bounds["lon_max"]
        )
