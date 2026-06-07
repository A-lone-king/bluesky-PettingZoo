"""STAR (Standard Terminal Arrival Route) approach scenario.

Aircraft follow STAR procedures with waypoint sequences, transitioning from
cruise to approach. This scenario tests:
- Multi-waypoint navigation with LNAV
- Speed and altitude management during descent
- Terminal airspace operations

The scenario uses 3 different STAR procedures converging on a single runway,
creating natural merge points for training conflict resolution.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.geometry import haversine_distance
from bluesky_pettingzoo.utils.types import ConflictConfig, SpawnConfig

# ============================================================
# STAR Procedure Definitions (Amsterdam Schiphol inspired)
# ============================================================
# Each STAR is a list of (name, lat, lon, alt_constraint) tuples
# alt_constraint: None = no constraint, float = cross altitude in feet

STAR_PROCEDURES = {
    "ARTIP3C": {
        "description": "ARTIP 3C arrival from East",
        "waypoints": [
            ("ARTIP", 52.8000, 5.5000, 24000.0),
            ("SULEG", 52.6000, 5.0000, 18000.0),
            ("AUDIK", 52.4000, 4.8000, 12000.0),
            ("BERGI", 52.2500, 4.7500, 8000.0),
            ("LOPIK", 52.1000, 4.8000, 6000.0),
        ],
        "transition_alt": 24000.0,
        "transition_speed": 280.0,
    },
    "RIVER4M": {
        "description": "RIVER 4M arrival from South",
        "waypoints": [
            ("RIVER", 51.5000, 4.8000, 24000.0),
            ("VEDON", 51.7000, 4.6000, 18000.0),
            ("HELEN", 51.9000, 4.5000, 12000.0),
            ("AMNOV", 52.0500, 4.6000, 8000.0),
            ("SUGOL", 52.1000, 4.7000, 6000.0),
        ],
        "transition_alt": 24000.0,
        "transition_speed": 280.0,
    },
    "SOBTU3G": {
        "description": "SOBTU 3G arrival from North",
        "waypoints": [
            ("SOBTU", 53.2000, 4.8000, 24000.0),
            ("IMDIK", 53.0000, 4.9000, 18000.0),
            ("NEKNU", 52.8000, 4.8500, 12000.0),
            ("PATCO", 52.5000, 4.8000, 8000.0),
            ("AVALI", 52.2000, 4.7500, 6000.0),
        ],
        "transition_alt": 24000.0,
        "transition_speed": 280.0,
    },
}

# Runway configuration (EHAM Runway 27)
RUNWAY_THRESHOLD = {
    "lat": 52.3080,
    "lon": 4.7639,
    "heading": 270.0,
}

# Approach fix (IAF - Initial Approach Fix)
APPROACH_FIXES = {
    "ARTIP3C": {"lat": 52.1000, "lon": 4.8000, "alt": 6000.0},
    "RIVER4M": {"lat": 52.1000, "lon": 4.7000, "alt": 6000.0},
    "SOBTU3G": {"lat": 52.1500, "lon": 4.7500, "alt": 6000.0},
}

# Final approach course (simplified ILS to Runway 27)
FINAL_APPROACH = {
    "fix": {"lat": 52.1000, "lon": 4.7500},
    "course": 270.0,
    "glide_slope": 3.0,  # degrees
    "decision_alt": 200.0,  # feet above threshold
}


class StarApproachScenario(BaseScenario):
    """STAR approach procedure scenario.

    Aircraft follow STAR procedures from cruise altitude to approach,
    testing multi-waypoint navigation, speed/altitude management,
    and merge operations in terminal airspace.

    Args:
        num_aircraft: Number of aircraft (max 3, one per STAR).
        seed: Optional seed for reproducibility.
    """

    def __init__(
        self,
        num_aircraft: int = 3,
        seed: int | None = None,
    ) -> None:
        self._num_aircraft = min(num_aircraft, 3)
        self._seed = seed
        self._agents: list[str] = []
        self._star_assignments: dict[str, str] = {}
        self._waypoint_indices: dict[str, int] = {}
        self._full_waypoints: dict[str, list[dict[str, float]]] = {}
        self._bounds: dict[str, float] = {}

    @property
    def control_mode(self) -> str:
        """SINGLE_RL: ego aircraft on one STAR, background on others."""
        return "SINGLE_RL"

    @property
    def ego_agent(self) -> str | None:
        """Return ego agent (first aircraft)."""
        return self._agents[0] if self._agents else None

    @property
    def background_agents(self) -> list[str]:
        """Return background agents (remaining aircraft)."""
        return self._agents[1:]

    def reset(self, rng: np.random.RandomState) -> None:
        """Reset scenario state."""
        self._agents = []
        self._star_assignments = {}
        self._waypoint_indices = {}
        self._full_waypoints = {}

    def setup(
        self,
        rng: np.random.RandomState,
        airspace_bounds: dict[str, float],
    ) -> list[str]:
        """Initialize STAR approach scenario.

        Assigns one aircraft per STAR procedure, starting at the first waypoint.

        Args:
            rng: Seeded random number generator.
            airspace_bounds: Dict with lat_min, lat_max, lon_min, lon_max.

        Returns:
            List of agent IDs.
        """
        self._bounds = airspace_bounds
        star_names = list(STAR_PROCEDURES.keys())[: self._num_aircraft]
        self._agents = [f"AC{i:03d}" for i in range(self._num_aircraft)]

        for i, acid in enumerate(self._agents):
            star_name = star_names[i]
            self._star_assignments[acid] = star_name
            self._waypoint_indices[acid] = 0

            # Build full waypoint list including approach fix and runway
            star = STAR_PROCEDURES[star_name]
            waypoints = []
            for name, lat, lon, alt in star["waypoints"]:
                waypoints.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "alt": alt or star["transition_alt"],
                        "name": name,
                    }
                )
            # Add approach fix
            fix = APPROACH_FIXES[star_name]
            waypoints.append(
                {
                    "lat": fix["lat"],
                    "lon": fix["lon"],
                    "alt": fix["alt"],
                    "name": f"{star_name}_IAF",
                }
            )
            # Add runway threshold
            waypoints.append(
                {
                    "lat": RUNWAY_THRESHOLD["lat"],
                    "lon": RUNWAY_THRESHOLD["lon"],
                    "alt": 0.0,
                    "name": "RWY27",
                }
            )
            self._full_waypoints[acid] = waypoints

        return list(self._agents)

    def get_spawn_config(self) -> SpawnConfig:
        """Spawn aircraft at STAR entry altitudes."""
        return SpawnConfig(
            altitude_range=(24000.0, 24000.0),
            speed_range=(280, 300),
            heading_range=(180, 360),
        )

    def get_conflict_config(self) -> ConflictConfig:
        """Standard conflict thresholds for terminal airspace."""
        return ConflictConfig(
            nmac_horizontal_nm=3.0,
            nmac_vertical_ft=1000.0,
            warning_horizontal_nm=5.0,
            warning_vertical_ft=2000.0,
        )

    def get_initial_positions(self) -> dict[str, tuple[float, float]] | None:
        """Place aircraft at STAR entry fixes."""
        positions = {}
        for acid in self._agents:
            star_name = self._star_assignments[acid]
            star = STAR_PROCEDURES[star_name]
            first_wp = star["waypoints"][0]
            # Start slightly before the first waypoint
            positions[acid] = (first_wp[1], first_wp[2])
        return positions

    def get_waypoint(self, agent_id: str) -> dict[str, float]:
        """Return the current waypoint for an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            Dict with lat, lon, alt, hdg keys.
        """
        idx = self._waypoint_indices.get(agent_id, 0)
        waypoints = self._full_waypoints.get(agent_id, [])
        if idx >= len(waypoints):
            idx = len(waypoints) - 1
        wp = waypoints[idx]

        # Calculate heading to next waypoint
        hdg = 0.0
        if idx + 1 < len(waypoints):
            next_wp = waypoints[idx + 1]
            hdg = self._calculate_heading(
                wp["lat"], wp["lon"], next_wp["lat"], next_wp["lon"]
            )

        return {
            "lat": wp["lat"],
            "lon": wp["lon"],
            "alt": wp["alt"],
            "hdg": hdg,
        }

    def update_waypoint(
        self,
        agent_id: str,
        state: Any,
    ) -> dict[str, float] | None:
        """Advance to next waypoint when current one is reached.

        Args:
            agent_id: Agent identifier.
            state: Current aircraft state.

        Returns:
            Next waypoint dict, or None if approach complete.
        """
        idx = self._waypoint_indices.get(agent_id, 0)
        waypoints = self._full_waypoints.get(agent_id, [])

        if idx + 1 >= len(waypoints):
            # Approach complete
            return None

        # Check if close enough to current waypoint (< 2 NM)
        current_wp = waypoints[idx]
        dist = haversine_distance(
            state.lat, state.lon, current_wp["lat"], current_wp["lon"]
        )
        if dist < 2.0:
            self._waypoint_indices[agent_id] = idx + 1
            next_wp = waypoints[idx + 1]
            return {
                "lat": next_wp["lat"],
                "lon": next_wp["lon"],
            }

        return None

    def configure_npc_navigation(self, wrapper: Any) -> list[str]:
        """Configure LNAV for all aircraft to follow their STAR routes.

        Args:
            wrapper: BlueSkyWrapper instance.

        Returns:
            List of BlueSky commands sent.
        """
        commands: list[str] = []

        for acid in self._agents:
            star_name = self._star_assignments[acid]
            star = STAR_PROCEDURES[star_name]
            waypoints = self._full_waypoints[acid]

            # Set origin at first waypoint
            first_wp = waypoints[0]
            wrapper.set_origin(acid, first_wp["lat"], first_wp["lon"])
            commands.append(f"ORIG {acid} {first_wp['lat']} {first_wp['lon']}")

            # Set destination at last waypoint (runway)
            last_wp = waypoints[-1]
            wrapper.set_destination(acid, last_wp["lat"], last_wp["lon"])
            commands.append(f"DEST {acid} {last_wp['lat']} {last_wp['lon']}")

            # Add intermediate waypoints (skip first and last)
            for wp in waypoints[1:-1]:
                wrapper.add_waypoint(acid, wp["lat"], wp["lon"])
                commands.append(f"ADDWPT {acid} {wp['lat']} {wp['lon']}")

            # Enable LNAV
            wrapper.enable_lnav(acid)
            commands.append(f"LNAV {acid} ON")

            # Set initial speed constraint
            wrapper.send_command(f"SPD {acid} {star['transition_speed']}")
            commands.append(f"SPD {acid} {star['transition_speed']}")

        return commands

    def get_priority(self, agent_id: str, state: Any) -> float:
        """Assign priority based on position in STAR.

        Aircraft closer to runway have higher priority.

        Args:
            agent_id: Agent identifier.
            state: Current aircraft state.

        Returns:
            Priority value (higher = higher priority).
        """
        idx = self._waypoint_indices.get(agent_id, 0)
        waypoints = self._full_waypoints.get(agent_id, [])
        total = len(waypoints) if waypoints else 1
        # Priority increases as aircraft progresses (0.0 to 1.0)
        return idx / total

    def should_truncate(
        self,
        agent_id: str,
        state: Any,
        airspace_bounds: dict[str, float],
    ) -> bool:
        """Truncate when aircraft lands (alt < 200 ft near runway).

        Args:
            agent_id: Agent identifier.
            state: Current aircraft state.
            airspace_bounds: Dict with lat_min, lat_max, lon_min, lon_max.

        Returns:
            True if agent should be truncated.
        """
        # Check if landed near runway
        dist_to_runway = haversine_distance(
            state.lat,
            state.lon,
            RUNWAY_THRESHOLD["lat"],
            RUNWAY_THRESHOLD["lon"],
        )
        if dist_to_runway < 1.0 and state.alt < 200.0:
            return True

        # Also truncate if leaving airspace
        return (
            state.lat < airspace_bounds["lat_min"]
            or state.lat > airspace_bounds["lat_max"]
            or state.lon < airspace_bounds["lon_min"]
            or state.lon > airspace_bounds["lon_max"]
        )

    def _calculate_heading(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate initial bearing between two points.

        Args:
            lat1, lon1: Start point.
            lat2, lon2: End point.

        Returns:
            Bearing in degrees (0-360).
        """
        import math

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)

        x = math.sin(dlon_rad) * math.cos(lat2_rad)
        y = (
            math.cos(lat1_rad) * math.sin(lat2_rad)
            - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
        )

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360
