"""Observation normalizer for converting raw values to [-1, 1] range."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class Normalizer:
    """Normalizes observation values to [-1, 1] range.

    Uses configuration parameters for mid/range values.
    All normalization formulas: (value - mid) / range
    Output is clipped to [-1, 1].
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize normalizer with configuration.

        Args:
            config: Configuration dictionary with normalization parameters
        """
        self.config = config
        norm_config = config.get("normalization", {})
        self._heading_mid = norm_config.get("heading", {}).get("mid", 180)
        self._heading_range = norm_config.get("heading", {}).get("range", 180)
        self._alt_mid = norm_config.get("altitude", {}).get("mid", 33000)
        self._alt_range = norm_config.get("altitude", {}).get("range", 10000)
        self._speed_mid = norm_config.get("speed", {}).get("mid", 450)
        self._speed_range = norm_config.get("speed", {}).get("range", 100)
        self._distance_max = norm_config.get("distance", {}).get("max", 20)
        self._lat_mid = norm_config.get("latitude", {}).get("mid", 0.0)
        self._lat_range = norm_config.get("latitude", {}).get("range", 90.0)
        self._lon_mid = norm_config.get("longitude", {}).get("mid", 0.0)
        self._lon_range = norm_config.get("longitude", {}).get("range", 180.0)
        self._vs_max = norm_config.get("vertical_speed", {}).get("max", 6000)

    def _clip(self, value: float) -> float:
        """Clip value to [-1, 1] range.

        Args:
            value: Value to clip

        Returns:
            Clipped value
        """
        return float(np.clip(value, -1.0, 1.0))

    def normalize_heading(self, heading: float) -> float:
        """Normalize heading to [-1, 1].

        Args:
            heading: Heading in degrees (0-360)

        Returns:
            Normalized heading
        """
        return self._clip((heading - self._heading_mid) / self._heading_range)

    def normalize_altitude(self, altitude: float) -> float:
        """Normalize altitude to [-1, 1].

        Args:
            altitude: Altitude in feet

        Returns:
            Normalized altitude
        """
        return self._clip((altitude - self._alt_mid) / self._alt_range)

    def normalize_speed(self, speed: float) -> float:
        """Normalize speed to [-1, 1].

        Args:
            speed: Speed in knots

        Returns:
            Normalized speed
        """
        return self._clip((speed - self._speed_mid) / self._speed_range)

    def normalize_lat(self, lat: float) -> float:
        """Normalize latitude to [-1, 1] relative to airspace center.

        Args:
            lat: Latitude in degrees

        Returns:
            Normalized latitude
        """
        return self._clip((lat - self._lat_mid) / self._lat_range)

    def normalize_lon(self, lon: float) -> float:
        """Normalize longitude to [-1, 1] relative to airspace center.

        Args:
            lon: Longitude in degrees

        Returns:
            Normalized longitude
        """
        return self._clip((lon - self._lon_mid) / self._lon_range)

    def normalize_vs(self, vs: float) -> float:
        """Normalize vertical speed to [-1, 1].

        Args:
            vs: Vertical speed in ft/min

        Returns:
            Normalized vertical speed
        """
        return self._clip(vs / self._vs_max)

    def normalize_distance(self, distance: float) -> float:
        """Normalize distance to [0, 1].

        Args:
            distance: Distance in nautical miles

        Returns:
            Normalized distance
        """
        return float(np.clip(distance / self._distance_max, 0.0, 1.0))

    def normalize_bearing(self, bearing: float) -> float:
        """Normalize bearing to [0, 1].

        Args:
            bearing: Bearing in degrees (0-360)

        Returns:
            Normalized bearing
        """
        return float(np.clip(bearing / 360.0, 0.0, 1.0))

    def normalize_heading_cos(self, heading: float) -> float:
        """Compute cos(heading) for circular continuity.

        Args:
            heading: Heading in degrees (0-360)

        Returns:
            cos(heading * pi / 180), in [-1, 1]
        """
        return math.cos(math.radians(heading))

    def normalize_heading_sin(self, heading: float) -> float:
        """Compute sin(heading) for circular continuity.

        Args:
            heading: Heading in degrees (0-360)

        Returns:
            sin(heading * pi / 180), in [-1, 1]
        """
        return math.sin(math.radians(heading))

    def normalize_bearing_cos(self, bearing: float) -> float:
        """Compute cos(bearing) for circular continuity.

        Args:
            bearing: Bearing in degrees (0-360)

        Returns:
            cos(bearing * pi / 180), in [-1, 1]
        """
        return math.cos(math.radians(bearing))

    def normalize_bearing_sin(self, bearing: float) -> float:
        """Compute sin(bearing) for circular continuity.

        Args:
            bearing: Bearing in degrees (0-360)

        Returns:
            sin(bearing * pi / 180), in [-1, 1]
        """
        return math.sin(math.radians(bearing))

    def normalize_aircraft_state(self, state: dict[str, Any]) -> dict[str, float]:
        """Normalize complete aircraft state.

        Args:
            state: Raw aircraft state dictionary

        Returns:
            Normalized state dictionary
        """
        return {
            "heading": self.normalize_heading(state["hdg"]),
            "altitude": self.normalize_altitude(state["alt"]),
            "speed": self.normalize_speed(state["tas"]),
            "lat": self.normalize_lat(state["lat"]),
            "lon": self.normalize_lon(state["lon"]),
            "vs": self.normalize_vs(state["vs"]),
        }

    def normalize_relative_position(
        self,
        distance_nm: float,
        bearing_deg: float,
    ) -> dict[str, float]:
        """Normalize relative position.

        Args:
            distance_nm: Distance in nautical miles
            bearing_deg: Bearing in degrees

        Returns:
            Normalized relative position
        """
        return {
            "distance": self.normalize_distance(distance_nm),
            "bearing": self.normalize_bearing(bearing_deg),
        }
