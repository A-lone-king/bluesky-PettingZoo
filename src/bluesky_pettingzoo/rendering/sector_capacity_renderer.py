"""SectorCapacity scenario renderer — sector boundary and capacity visualization."""

from __future__ import annotations

from typing import Any

from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer
from bluesky_pettingzoo.rendering.common import (
    draw_aircraft,
    draw_nmac_circle,
    draw_sector_polygon,
    draw_waypoint,
    latlon_to_pixel,
)


class SectorCapacityRenderer(BaseRenderer):
    """Renderer for SectorCapacity scenario.

    Draws sector boundaries as rectangles with capacity labels,
    aircraft, and waypoints.
    """

    def __init__(self, width: int = 800, height: int = 600) -> None:
        super().__init__(width=width, height=height, caption="SectorCapacity")
        self._bounds: dict[str, float] = {}

    def set_bounds(self, bounds: dict[str, float]) -> None:
        """Set the lat/lon bounds for coordinate conversion."""
        self._bounds = bounds

    def render_frame(
        self,
        states: dict[str, Any],
        waypoints: dict[str, dict[str, float]] | None = None,
        step: int = 0,
        info: dict[str, Any] | None = None,
    ) -> None:
        if not self._initialized or self._screen is None:
            return

        self._screen.fill((0, 0, 0))
        bounds = self._bounds or {
            "lat_min": 39.0,
            "lat_max": 41.0,
            "lon_min": 116.0,
            "lon_max": 118.0,
        }

        # Draw sector boundaries
        sectors = info.get("sectors", []) if info else []
        for sector in sectors:
            sector_bounds = sector.get("bounds")
            if sector_bounds and len(sector_bounds) == 2:
                (lat_min, lon_min), (lat_max, lon_max) = sector_bounds
                vertices = [
                    (lat_min, lon_min),
                    (lat_min, lon_max),
                    (lat_max, lon_max),
                    (lat_max, lon_min),
                ]
                draw_sector_polygon(
                    self._screen,
                    vertices,
                    bounds,
                    self._width,
                    self._height,
                    color=(100, 100, 255),
                    line_width=2,
                )
                # Draw capacity label at sector center
                if self._font is not None:
                    center_lat = (lat_min + lat_max) / 2
                    center_lon = (lon_min + lon_max) / 2
                    cx, cy = latlon_to_pixel(
                        center_lat,
                        center_lon,
                        bounds,
                        self._width,
                        self._height,
                    )
                    sid = sector.get("id", "")
                    cap = sector.get("capacity", "?")
                    # Count aircraft in this sector
                    count = sum(
                        1
                        for s in states.values()
                        if lat_min <= s.lat <= lat_max and lon_min <= s.lon <= lon_max
                    )
                    label = f"{sid}: {count}/{cap}"
                    cap_text = self._font.render(label, True, (200, 200, 255))
                    self._screen.blit(cap_text, (cx - 30, cy - 10))

        for acid, state in states.items():
            x, y = latlon_to_pixel(state.lat, state.lon, bounds, self._width, self._height)
            draw_aircraft(self._screen, x, y, state.hdg)
            draw_nmac_circle(self._screen, x, y)

        if waypoints and isinstance(waypoints, dict):
            for acid, wp in waypoints.items():
                if isinstance(wp, dict) and "lat" in wp:
                    wx, wy = latlon_to_pixel(
                        wp["lat"],
                        wp["lon"],
                        bounds,
                        self._width,
                        self._height,
                    )
                    draw_waypoint(self._screen, wx, wy)

        self._draw_hud(step=step, info=info)
        self.flip()
