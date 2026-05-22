"""Observation manager — integrates normalizer and filters."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from bluesky_pettingzoo.observations.filters import PerceptionFilter
from bluesky_pettingzoo.observations.normalizer import Normalizer
from bluesky_pettingzoo.utils.types import AircraftState


class ObservationManager:
    """Generates normalized observations for RL agents.

    Combines perception filtering and value normalization to produce
    Dict-format observations, textual state, and airspace snapshots.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._normalizer = Normalizer(config)
        self._filter = PerceptionFilter(config)
        obs = config.get("observation", {})
        self._max_obs: int = obs.get("max_observable_aircraft", 10)

    def observation_space(self) -> spaces.Dict:
        """Return the gymnasium observation space definition.

        self_state layout: [heading_cos, heading_sin, altitude, speed, lat, lon, vs, ground_speed]
        other_aircraft layout: [heading, altitude, speed, distance, bearing_cos, bearing_sin, lat, lon, relative_altitude]
        """
        low = np.array([-1.0, -1.0, -1.0, -1.0, -90.0, -180.0, -6000.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 90.0, 180.0, 6000.0, 1.0], dtype=np.float32)
        other_low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -90.0, -180.0, -1.0], dtype=np.float32)
        other_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 90.0, 180.0, 1.0], dtype=np.float32)
        return spaces.Dict({
            "self_state": spaces.Box(low=low, high=high, dtype=np.float32),
            "other_aircraft": spaces.Box(
                low=np.tile(other_low, (self._max_obs, 1)),
                high=np.tile(other_high, (self._max_obs, 1)),
                dtype=np.float32,
            ),
            "other_aircraft_mask": spaces.Box(
                low=0, high=1, shape=(self._max_obs,), dtype=np.int8,
            ),
            "goal": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        })

    def generate(
        self,
        own_state: AircraftState,
        other_states: list[AircraftState],
        goal: dict[str, float],
        conflict_status: str = "safe",
        airspace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate complete observation package.

        Args:
            own_state: Ownship aircraft state.
            other_states: All other aircraft states (pre-filter).
            goal: Goal waypoint {lat, lon, alt, hdg}.
            conflict_status: Current conflict status string.
            airspace: Optional airspace topology data.

        Returns:
            Dict with keys: observation, textual_state, airspace_snapshot.
        """
        # Filter and get observable aircraft
        filtered = self._filter.filter(own_state, other_states)

        # Build self_state: [heading_cos, heading_sin, altitude, speed, lat, lon, vs, ground_speed]
        norm_self = self._normalizer.normalize_aircraft_state(own_state)
        self_state = np.array([
            self._normalizer.normalize_heading_cos(own_state["hdg"]),
            self._normalizer.normalize_heading_sin(own_state["hdg"]),
            norm_self["altitude"],
            norm_self["speed"],
            norm_self["lat"],
            norm_self["lon"],
            norm_self["vs"],
            norm_self["speed"],  # ground_speed ≈ tas for now
        ], dtype=np.float32)

        # Build other_aircraft array: [heading, altitude, speed, distance, bearing_cos, bearing_sin, relative_altitude, relative_speed_x, relative_speed_y]
        import math
        other_aircraft = np.zeros((self._max_obs, 9), dtype=np.float32)
        mask = np.zeros(self._max_obs, dtype=np.int8)

        # Own ship velocity components (north/east)
        own_vx = own_state["tas"] * math.sin(math.radians(own_state["hdg"]))
        own_vy = own_state["tas"] * math.cos(math.radians(own_state["hdg"]))

        for i, entry in enumerate(filtered[: self._max_obs]):
            st = entry["state"]
            norm = self._normalizer.normalize_aircraft_state(st)
            rel = self._normalizer.normalize_relative_position(
                entry["distance_nm"], entry["bearing_deg"],
            )
            # Other aircraft velocity components
            other_vx = st["tas"] * math.sin(math.radians(st["hdg"]))
            other_vy = st["tas"] * math.cos(math.radians(st["hdg"]))
            # Relative speed (normalized by max speed range)
            max_speed = self._normalizer._speed_mid + self._normalizer._speed_range
            rel_speed_x = (other_vx - own_vx) / max_speed
            rel_speed_y = (other_vy - own_vy) / max_speed

            other_aircraft[i] = [
                norm["heading"],
                norm["altitude"],
                norm["speed"],
                rel["distance"],
                self._normalizer.normalize_bearing_cos(entry["bearing_deg"]),
                self._normalizer.normalize_bearing_sin(entry["bearing_deg"]),
                self._normalizer.normalize_altitude(st["alt"]) - norm_self["altitude"],
                float(np.clip(rel_speed_x, -1.0, 1.0)),
                float(np.clip(rel_speed_y, -1.0, 1.0)),
            ]
            mask[i] = 1

        # Build goal: [distance, bearing_cos, bearing_sin, alt_diff]
        from bluesky_pettingzoo.utils.geometry import haversine_distance, bearing

        goal_dist = haversine_distance(
            own_state["lat"], own_state["lon"], goal["lat"], goal["lon"],
        )
        goal_bear = bearing(
            own_state["lat"], own_state["lon"], goal["lat"], goal["lon"],
        )
        goal_vec = np.array([
            self._normalizer.normalize_distance(goal_dist),
            self._normalizer.normalize_bearing_cos(goal_bear),
            self._normalizer.normalize_bearing_sin(goal_bear),
            self._normalizer.normalize_altitude(goal["alt"]) - self._normalizer.normalize_altitude(own_state["alt"]),
        ], dtype=np.float32)

        observation = {
            "self_state": self_state,
            "other_aircraft": other_aircraft,
            "other_aircraft_mask": mask,
            "goal": goal_vec,
        }

        # Build textual state
        observable_list = [
            {
                "id": entry["state"].id,
                "distance_nm": round(entry["distance_nm"], 2),
                "bearing_deg": round(entry["bearing_deg"], 1),
                "altitude": entry["state"].alt,
            }
            for entry in filtered
        ]
        text = (
            f"Aircraft {own_state.id} at lat={own_state.lat:.4f}, "
            f"lon={own_state.lon:.4f}, alt={own_state.alt:.0f}ft, "
            f"hdg={own_state.hdg:.0f}, tas={own_state.tas:.0f}kt. "
            f"Observable: {len(filtered)} aircraft. "
            f"Conflict: {conflict_status}."
        )
        textual_state = {
            "agent_id": own_state.id,
            "position": {"lat": own_state.lat, "lon": own_state.lon},
            "heading": own_state.hdg,
            "altitude": own_state.alt,
            "speed": own_state.tas,
            "observable_aircraft": observable_list,
            "conflict_status": conflict_status,
            "text": text,
        }

        # Build airspace snapshot
        all_aircraft = [own_state] + other_states
        aircraft_positions = {
            s.id: {"lat": s.lat, "lon": s.lon, "alt": s.alt}
            for s in all_aircraft
        }
        airspace_snapshot = {
            "sectors": (airspace or {}).get("sectors", []),
            "waypoints": (airspace or {}).get("waypoints", []),
            "aircraft_positions": aircraft_positions,
        }

        return {
            "observation": observation,
            "textual_state": textual_state,
            "airspace_snapshot": airspace_snapshot,
        }
