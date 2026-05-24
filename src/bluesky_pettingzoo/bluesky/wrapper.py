"""BlueSky simulator wrapper for headless operation."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import bluesky as bs
except ImportError:
    bs = None  # type: ignore[assignment]


class BlueSkyWrapper:
    """Wrapper for BlueSky simulator in headless mode.

    Provides a clean Python API for interacting with BlueSky
    without GUI dependencies.

    Requires the ``bluesky`` package. Install with::

        pip install bluesky-pettingzoo[bluesky]
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the wrapper.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.dt: float = config["simulation"]["dt"]
        self._initialized: bool = False
        self._managed_aircraft: set[str] = set()
        self._airspace_bounds: dict[str, tuple[float, float]] = {}
        self._parse_airspace_bounds()

    def _parse_airspace_bounds(self) -> None:
        """Parse airspace bounds from config."""
        sectors = self.config.get("airspace", {}).get("sectors", [])
        if not sectors:
            return

        all_lats: list[float] = []
        all_lons: list[float] = []
        for sector in sectors:
            bounds = sector.get("bounds", [[0, 0], [0, 0]])
            all_lats.extend([bounds[0][0], bounds[1][0]])
            all_lons.extend([bounds[0][1], bounds[1][1]])

        self._airspace_bounds = {
            "lat_min": min(all_lats),
            "lat_max": max(all_lats),
            "lon_min": min(all_lons),
            "lon_max": max(all_lons),
        }

    def init_simulation(self) -> None:
        """Initialize BlueSky in headless mode.

        Raises:
            ImportError: If the bluesky package is not installed.
        """
        if self._initialized:
            return
        if bs is None:
            raise ImportError(
                "bluesky is required for real simulation. "
                "Install with: pip install bluesky-pettingzoo[bluesky]"
            )
        bs.init(mode="sim", detached=True)
        # Set simulation timestep from config
        bs.stack.stack(f"DT {self.dt};FF")
        # Disable built-in conflict resolution so agent commands are not overridden
        bs.stack.stack("reso off")
        self._initialized = True

    def step(self) -> float:
        """Advance simulation by one timestep.

        Returns:
            Current simulation time
        """
        return self.step_n(1)

    def step_n(self, n: int) -> float:
        """Advance simulation by n timesteps.

        Args:
            n: Number of simulation steps to execute

        Returns:
            Current simulation time
        """
        for _ in range(n):
            bs.sim.step()
        return float(bs.sim.simt)

    def reset(self) -> None:
        """Reset the simulation by deleting all aircraft."""
        if not self._initialized:
            return
        for acid in list(bs.traf.id):
            bs.stack.stack(f"DELETE {acid}")
        bs.sim.step()

    def create_aircraft(
        self,
        acid: str,
        actype: str,
        lat: float,
        lon: float,
        alt: float,
        hdg: float,
        spd: float,
    ) -> None:
        """Create a new aircraft.

        Args:
            acid: Aircraft ID
            actype: Aircraft type
            lat: Latitude
            lon: Longitude
            alt: Altitude (feet)
            hdg: Heading (degrees)
            spd: True airspeed (knots)
        """
        bs.traf.cre(acid, actype, lat, lon, hdg, alt, spd)
        self._managed_aircraft.add(acid)

    def remove_aircraft(self, acid: str) -> None:
        """Remove an aircraft.

        Args:
            acid: Aircraft ID to remove
        """
        bs.stack.stack(f"DELETE {acid}")
        self._managed_aircraft.discard(acid)

    def send_command(self, command: str) -> None:
        """Send a single command to BlueSky.

        Args:
            command: BlueSky command string
        """
        bs.stack.stack(command)

    def send_commands_batch(self, commands: list[str]) -> None:
        """Send multiple commands to BlueSky.

        Args:
            commands: List of command strings
        """
        for cmd in commands:
            bs.stack.stack(cmd)

    def _resolve_idx(self, acid: str) -> int:
        """Resolve aircraft ID to index.

        Args:
            acid: Aircraft ID

        Returns:
            Index in traffic array, or -1 if not found
        """
        idx = bs.traf.id2idx(acid)
        if isinstance(idx, np.ndarray):
            return int(idx[0]) if len(idx) > 0 else -1
        if isinstance(idx, (list, tuple)):
            return int(idx[0]) if len(idx) > 0 else -1
        return int(idx)

    def get_aircraft_state(self, acid: str) -> dict[str, Any]:
        """Get state of a single aircraft.

        Args:
            acid: Aircraft ID

        Returns:
            Dictionary with aircraft state

        Raises:
            ValueError: If aircraft not found
        """
        idx = self._resolve_idx(acid)
        if idx < 0:
            raise ValueError(f"Aircraft {acid} not found")

        return {
            "id": str(bs.traf.id[idx]),
            "lat": float(bs.traf.lat[idx]),
            "lon": float(bs.traf.lon[idx]),
            "alt": float(bs.traf.alt[idx]),
            "hdg": float(bs.traf.hdg[idx]),
            "tas": float(bs.traf.tas[idx]),
            "vs": float(bs.traf.vs[idx]),
        }

    def get_all_aircraft_states(self) -> dict[str, dict[str, Any]]:
        """Get states of all aircraft.

        Returns:
            Dictionary mapping aircraft IDs to their states
        """
        states: dict[str, dict[str, Any]] = {}
        for i in range(len(bs.traf.id)):
            acid = str(bs.traf.id[i])
            states[acid] = {
                "id": acid,
                "lat": float(bs.traf.lat[i]),
                "lon": float(bs.traf.lon[i]),
                "alt": float(bs.traf.alt[i]),
                "hdg": float(bs.traf.hdg[i]),
                "tas": float(bs.traf.tas[i]),
                "vs": float(bs.traf.vs[i]),
            }
        return states

    def get_active_aircraft_ids(self) -> list[str]:
        """Get list of active aircraft IDs.

        Returns:
            List of aircraft ID strings
        """
        return [str(acid) for acid in bs.traf.id]

    def is_aircraft_in_airspace(self, acid: str) -> bool:
        """Check if aircraft is within airspace bounds.

        Args:
            acid: Aircraft ID

        Returns:
            True if aircraft is in airspace
        """
        if not self._airspace_bounds:
            return True

        idx = self._resolve_idx(acid)
        if idx < 0:
            return False

        lat = float(bs.traf.lat[idx])
        lon = float(bs.traf.lon[idx])

        return (
            self._airspace_bounds["lat_min"] <= lat <= self._airspace_bounds["lat_max"]
            and self._airspace_bounds["lon_min"] <= lon <= self._airspace_bounds["lon_max"]
        )

    def close(self) -> None:
        """Close the simulator by removing managed aircraft."""
        if self._initialized:
            for acid in list(self._managed_aircraft):
                try:
                    bs.stack.stack(f"DELETE {acid}")
                except Exception:
                    pass
            self._managed_aircraft.clear()
            try:
                bs.sim.step()
            except Exception:
                pass
        self._initialized = False
