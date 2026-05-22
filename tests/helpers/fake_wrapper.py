"""In-memory BlueSkyWrapper for testing without real BlueSky."""

from __future__ import annotations

import math
from typing import Any


class FakeBlueSkyWrapper:
    """Drop-in replacement for BlueSkyWrapper that uses in-memory state.

    Simulates aircraft movement using heading and speed so that
    integration tests can run without the real BlueSky simulator.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.dt: float = config["simulation"]["dt"]
        self._initialized = False
        self._simt: float = 0.0
        self._step_count: int = 0
        self._aircraft: dict[str, dict[str, Any]] = {}

        bounds_list = config.get("airspace", {}).get("sectors", [])
        if bounds_list:
            lats = [b["bounds"][0][0] for b in bounds_list] + [b["bounds"][1][0] for b in bounds_list]
            lons = [b["bounds"][0][1] for b in bounds_list] + [b["bounds"][1][1] for b in bounds_list]
            self._bounds = {
                "lat_min": min(lats), "lat_max": max(lats),
                "lon_min": min(lons), "lon_max": max(lons),
            }
        else:
            self._bounds = {}

    def init_simulation(self) -> None:
        self._initialized = True

    def step(self) -> float:
        return self.step_n(1)

    def step_n(self, n: int) -> float:
        for _ in range(n):
            self._step_count += 1
            self._simt += self.dt
            for st in self._aircraft.values():
                spd_nm_s = st["tas"] / 3600.0
                hdg_rad = math.radians(st["hdg"])
                st["lat"] += math.cos(hdg_rad) * spd_nm_s * self.dt / 60.0
                st["lon"] += math.sin(hdg_rad) * spd_nm_s * self.dt / (
                    60.0 * math.cos(math.radians(st["lat"]))
                )
        return self._simt

    def reset(self) -> None:
        self._aircraft.clear()
        self._simt = 0.0

    def create_aircraft(
        self, acid: str, actype: str, lat: float, lon: float,
        alt: float, hdg: float, spd: float,
    ) -> None:
        self._aircraft[acid] = {
            "id": acid, "lat": lat, "lon": lon,
            "alt": alt, "hdg": hdg, "tas": spd, "vs": 0.0,
        }

    def remove_aircraft(self, acid: str) -> None:
        self._aircraft.pop(acid, None)

    def send_command(self, command: str) -> None:
        pass

    def send_commands_batch(self, commands: list[str]) -> None:
        pass

    def get_aircraft_state(self, acid: str) -> dict[str, Any]:
        if acid not in self._aircraft:
            raise ValueError(f"Aircraft {acid} not found")
        return dict(self._aircraft[acid])

    def get_all_aircraft_states(self) -> dict[str, dict[str, Any]]:
        return {k: dict(v) for k, v in self._aircraft.items()}

    def get_active_aircraft_ids(self) -> list[str]:
        return list(self._aircraft.keys())

    def is_aircraft_in_airspace(self, acid: str) -> bool:
        if acid not in self._aircraft:
            return False
        if not self._bounds:
            return True
        st = self._aircraft[acid]
        return (
            self._bounds["lat_min"] <= st["lat"] <= self._bounds["lat_max"]
            and self._bounds["lon_min"] <= st["lon"] <= self._bounds["lon_max"]
        )

    def close(self) -> None:
        self._initialized = False
