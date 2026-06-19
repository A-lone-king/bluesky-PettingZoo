"""Observation manager — integrates normalizer and filters."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from gymnasium import spaces

from bluesky_pettingzoo.observations.filters import PerceptionFilter
from bluesky_pettingzoo.observations.normalizer import Normalizer
from bluesky_pettingzoo.utils.geometry import bearing, haversine_distance
from bluesky_pettingzoo.utils.types import AircraftState


class ObservationManager:
    """Generates normalized observations for RL agents.

    Combines perception filtering and value normalization to produce
    Dict-format observations, textual state, and airspace snapshots.

    When ``max_obstacles > 0``, the observation space includes an
    optional ``obstacles`` key with per-obstacle distance, bearing,
    and radius information.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._normalizer = Normalizer(config)
        self._filter = PerceptionFilter(config)
        obs = config.get("observation", {})
        self._max_obs: int = obs.get("max_observable_aircraft", 10)
        self._max_obstacles: int = obs.get("max_obstacles", 0)

    def observation_space(self) -> spaces.Dict:
        """Return the gymnasium observation space definition.

        self_state layout: [heading_cos, heading_sin, altitude, speed,
            lat, lon, vs, ground_speed, priority]
        other_aircraft layout: [heading, altitude, speed, distance, bearing_cos, bearing_sin,
            relative_altitude, relative_speed_x, relative_speed_y, priority,
            time_to_conflict, closure_rate]
        obstacles layout (per obstacle): [distance, bearing_cos, bearing_sin, radius]
        """
        low = np.full(9, -1.0, dtype=np.float32)
        high = np.full(9, 1.0, dtype=np.float32)
        other_low = np.full(12, -1.0, dtype=np.float32)
        other_high = np.full(12, 1.0, dtype=np.float32)
        space_dict: dict[str, spaces.Space[Any]] = {
            "self_state": spaces.Box(low=low, high=high, dtype=np.float32),
            "other_aircraft": spaces.Box(
                low=np.tile(other_low, (self._max_obs, 1)),
                high=np.tile(other_high, (self._max_obs, 1)),
                dtype=np.float32,
            ),
            "other_aircraft_mask": spaces.Box(
                low=0,
                high=1,
                shape=(self._max_obs,),
                dtype=np.int8,
            ),
            "goal": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            "conflict_state": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
        }
        if self._max_obstacles > 0:
            obs_feat_low = np.array([0.0, -1.0, -1.0, 0.0], dtype=np.float32)
            obs_feat_high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            space_dict["obstacles"] = spaces.Dict(
                {
                    "position": spaces.Box(
                        low=np.tile(obs_feat_low, (self._max_obstacles, 1)),
                        high=np.tile(obs_feat_high, (self._max_obstacles, 1)),
                        dtype=np.float32,
                    ),
                    "mask": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self._max_obstacles,),
                        dtype=np.int8,
                    ),
                }
            )
        return spaces.Dict(space_dict)

    def _encode_conflict_status(self, status: str) -> np.ndarray:
        """Encode conflict status as one-hot vector.

        Args:
            status: Conflict status string ("nmac", "warning", or "safe").

        Returns:
            One-hot vector of shape (3,): [is_nmac, is_warning, is_safe].
        """
        vec = np.zeros(3, dtype=np.float32)
        if status == "nmac":
            vec[0] = 1.0
        elif status == "warning":
            vec[1] = 1.0
        else:
            vec[2] = 1.0
        return vec

    def generate(
        self,
        own_state: AircraftState,
        other_states: list[AircraftState],
        goal: dict[str, float],
        conflict_status: str = "safe",
        airspace: dict[str, Any] | None = None,
        obstacle_polygons: list[list[tuple[float, float]]] | None = None,
        agent_priorities: dict[str, float] | None = None,
        waypoints: list[dict[str, float]] | None = None,
        waypoints_reached: list[bool] | None = None,
    ) -> dict[str, Any]:
        """Generate complete observation package.

        Args:
            own_state: Ownship aircraft state.
            other_states: All other aircraft states (pre-filter).
            goal: Goal waypoint {lat, lon, alt, hdg}.
            conflict_status: Current conflict status string.
            airspace: Optional airspace topology data.
            obstacle_polygons: Optional list of obstacle polygons, each a
                list of (lat, lon) vertices.
            agent_priorities: Optional dict mapping agent ID to priority value
                (normalized [-1, 1]).

        Returns:
            Dict with keys: observation, textual_state, airspace_snapshot.
        """
        # Filter and get observable aircraft
        filtered = self._filter.filter(own_state, other_states)

        # Build self_state: [heading_cos, heading_sin, altitude, speed,
        #     lat, lon, vs, ground_speed, priority]
        norm_self = self._normalizer.normalize_aircraft_state(own_state)  # type: ignore[arg-type]
        own_priority = float(np.clip((agent_priorities or {}).get(own_state.id, 0.0), -1.0, 1.0))

        # Compute ground speed (TAS + wind effect)
        # ground_speed = sqrt((tas*sin(hdg) + wind_east)^2 + (tas*cos(hdg) + wind_north)^2)
        # For now, use TAS as ground speed (no wind field support)
        # TODO: Integrate wind field data when available
        ground_speed = own_state["tas"]
        ground_speed_norm = self._normalizer.normalize_speed(ground_speed)

        self_state = np.array(
            [
                self._normalizer.normalize_heading_cos(own_state["hdg"]),
                self._normalizer.normalize_heading_sin(own_state["hdg"]),
                norm_self["altitude"],
                norm_self["speed"],
                norm_self["lat"],
                norm_self["lon"],
                norm_self["vs"],
                ground_speed_norm,
                own_priority,
            ],
            dtype=np.float32,
        )

        # Build other_aircraft array: [heading, altitude, speed, distance, bearing_cos,
        # bearing_sin, relative_altitude, relative_speed_x, relative_speed_y, priority,
        # time_to_conflict, closure_rate]
        other_aircraft = np.zeros((self._max_obs, 12), dtype=np.float32)
        mask = np.zeros(self._max_obs, dtype=np.int8)

        # Own ship velocity components (north/east)
        own_vx = own_state["tas"] * math.sin(math.radians(own_state["hdg"]))
        own_vy = own_state["tas"] * math.cos(math.radians(own_state["hdg"]))

        for i, entry in enumerate(filtered[: self._max_obs]):
            st = entry["state"]
            norm = self._normalizer.normalize_aircraft_state(st)
            rel = self._normalizer.normalize_relative_position(
                entry["distance_nm"],
                entry["bearing_deg"],
            )
            # Other aircraft velocity components
            other_vx = st["tas"] * math.sin(math.radians(st["hdg"]))
            other_vy = st["tas"] * math.cos(math.radians(st["hdg"]))
            # Relative speed (normalized by max speed range)
            max_speed = self._normalizer._speed_mid + self._normalizer._speed_range
            rel_speed_x = (other_vx - own_vx) / max_speed
            rel_speed_y = (other_vy - own_vy) / max_speed

            # Compute conflict prediction features
            # Closure rate: component of relative velocity along line-of-sight
            # Positive closure = closing (distance decreasing), negative = opening
            bear_rad = math.radians(entry["bearing_deg"])
            # Unit vector from own to other (in north-east frame)
            los_n = math.cos(bear_rad)
            los_e = math.sin(bear_rad)
            # Relative velocity of other wrt own in north-east frame
            rel_vn = other_vy - own_vy  # north component
            rel_ve = other_vx - own_vx  # east component
            # Closure rate: projection of -relative_velocity onto line-of-sight
            # (because closing means distance decreasing)
            closure = -(rel_vn * los_n + rel_ve * los_e)
            closure_normalized = float(np.clip(closure / max_speed, -1.0, 1.0))

            # Time to conflict (CPA): distance / closure rate
            # Positive closure means closing, negative means opening
            if closure > 1.0:  # Closing (at least 1 NM/s)
                ttc = entry["distance_nm"] / closure  # in seconds
                # Normalize: 0 = immediate conflict, 1 = far future
                # Use 300 seconds (5 min) as reference
                ttc_normalized = float(np.clip(1.0 - ttc / 300.0, -1.0, 1.0))
            else:
                # Not closing or opening significantly
                ttc_normalized = -1.0  # No conflict predicted

            priority = (agent_priorities or {}).get(st.id, 0.0)
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
                float(np.clip(priority, -1.0, 1.0)),
                ttc_normalized,
                closure_normalized,
            ]
            mask[i] = 1

        # Build goal: [distance, bearing_cos, bearing_sin, alt_diff]
        goal_dist = haversine_distance(
            own_state["lat"],
            own_state["lon"],
            goal["lat"],
            goal["lon"],
        )
        goal_bear = bearing(
            own_state["lat"],
            own_state["lon"],
            goal["lat"],
            goal["lon"],
        )
        goal_vec = np.array(
            [
                self._normalizer.normalize_distance(goal_dist),
                self._normalizer.normalize_bearing_cos(goal_bear),
                self._normalizer.normalize_bearing_sin(goal_bear),
                self._normalizer.normalize_altitude(goal["alt"])
                - self._normalizer.normalize_altitude(own_state["alt"]),
            ],
            dtype=np.float32,
        )

        # Build obstacles observation (optional)
        obstacles_obs = None
        if self._max_obstacles > 0 and obstacle_polygons:
            obs_position = np.zeros((self._max_obstacles, 4), dtype=np.float32)
            obs_mask = np.zeros(self._max_obstacles, dtype=np.int8)
            for i, polygon in enumerate(obstacle_polygons[: self._max_obstacles]):
                # Compute centroid of the polygon
                c_lat = sum(v[0] for v in polygon) / len(polygon)
                c_lon = sum(v[1] for v in polygon) / len(polygon)
                dist = haversine_distance(own_state["lat"], own_state["lon"], c_lat, c_lon)
                bear = bearing(own_state["lat"], own_state["lon"], c_lat, c_lon)
                # Approximate radius as max distance from centroid to any vertex
                radius_nm = max(haversine_distance(c_lat, c_lon, v[0], v[1]) for v in polygon)
                # Normalize radius by perception radius (default 20 NM)
                max_dist = self.config.get("observation", {}).get("perception_radius_nm", 20)
                obs_position[i] = [
                    self._normalizer.normalize_distance(dist),
                    self._normalizer.normalize_bearing_cos(bear),
                    self._normalizer.normalize_bearing_sin(bear),
                    float(np.clip(radius_nm / max_dist, 0.0, 1.0)),
                ]
                obs_mask[i] = 1
            obstacles_obs = {
                "position": obs_position,
                "mask": obs_mask,
            }

        observation = {
            "self_state": self_state,
            "other_aircraft": other_aircraft,
            "other_aircraft_mask": mask,
            "goal": goal_vec,
            "conflict_state": self._encode_conflict_status(conflict_status),
        }
        if obstacles_obs is not None:
            observation["obstacles"] = obstacles_obs

        # Build waypoints observation (for PlanWaypoint scenario)
        if waypoints is not None:
            num_wp = len(waypoints)
            wp_features = np.zeros((num_wp, 4), dtype=np.float32)
            wp_mask = np.zeros(num_wp, dtype=np.int8)
            for i, wp in enumerate(waypoints):
                dist = haversine_distance(
                    own_state["lat"],
                    own_state["lon"],
                    wp["lat"],
                    wp["lon"],
                )
                bear = bearing(
                    own_state["lat"],
                    own_state["lon"],
                    wp["lat"],
                    wp["lon"],
                )
                reached = waypoints_reached[i] if waypoints_reached is not None else False
                wp_features[i] = [
                    self._normalizer.normalize_distance(dist),
                    self._normalizer.normalize_bearing_cos(bear),
                    self._normalizer.normalize_bearing_sin(bear),
                    1.0 if reached else 0.0,
                ]
                wp_mask[i] = 0 if reached else 1
            observation["waypoints"] = {
                "features": wp_features,
                "mask": wp_mask,
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
            s.id: {"lat": s.lat, "lon": s.lon, "alt": s.alt} for s in all_aircraft
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
