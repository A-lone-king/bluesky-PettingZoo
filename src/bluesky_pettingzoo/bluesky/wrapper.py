"""BlueSky simulator wrapper for headless operation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import bluesky as bs
    from bluesky.tools.aero import vtas2cas
except ImportError:
    bs = None
    vtas2cas = None


_FT_TO_M = 0.3048
_KTS_TO_MS = 1852.0 / 3600.0
_bs_global_initialized = False


class BlueSkyWrapper:
    """Wrapper for BlueSky simulator in headless mode.

    Provides a clean Python API for interacting with BlueSky
    without GUI dependencies.

    The public interface uses **feet** for altitude, **knots** for speed,
    and **ft/min** for vertical speed.  BlueSky's internal arrays use SI
    (meters and m/s), so conversions are applied in ``create_aircraft`` /
    ``get_aircraft_state`` / ``get_all_aircraft_states``.

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
        self._step_count: int = 0
        self._simt: float = 0.0
        self._managed_aircraft: set[str] = set()
        self._airspace_bounds: dict[str, float] = {}
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
        global _bs_global_initialized
        if bs is None:
            raise ImportError(
                "bluesky is required for real simulation. "
                "Install with: pip install bluesky-pettingzoo[bluesky]"
            )
        if not _bs_global_initialized:
            bs.init(mode="sim", detached=True)
            _bs_global_initialized = True
        # Reset simulation state for this wrapper instance
        try:
            bs.sim.reset()
        except Exception:
            # If reset fails, re-initialize from scratch
            bs.init(mode="sim", detached=True)
            bs.sim.reset()
        bs.stack.stack(f"DT {self.dt};FF")
        bs.stack.stack("reso off")

        # Activate performance model (openap, bada, or off)
        perf_model = self.config.get("simulation", {}).get("performance_model", "openap")
        if perf_model and perf_model.lower() != "off":
            bs.stack.stack(f"PERF {perf_model}")

        self._initialized = True
        self._step_count = 0
        self._simt = 0.0

    def step(self, on_substep: Callable[[int], bool] | None = None) -> float:
        """Advance simulation by one timestep.

        Args:
            on_substep: Optional callback invoked after each substep.
                Receives 0-based step index. Return True to continue,
                False to stop early.

        Returns:
            Current simulation time
        """
        return self.step_n(1, on_substep=on_substep)

    def step_n(
        self,
        n: int,
        on_substep: Callable[[int], bool] | None = None,
    ) -> float:
        """Advance simulation by n timesteps.

        Args:
            n: Number of simulation steps to execute
            on_substep: Optional callback invoked after each substep.
                Receives 0-based step index. Return True to continue,
                False to stop early.

        Returns:
            Current simulation time
        """
        for i in range(n):
            bs.sim.step()
            self._step_count += 1
            if on_substep is not None and not on_substep(i):
                break
        self._simt = float(bs.sim.simt)
        return self._simt

    def reset(self) -> None:
        """Reset the simulation by deleting all aircraft and resetting the clock."""
        if not self._initialized:
            return
        bs.sim.reset()
        bs.stack.stack(f"DT {self.dt};FF")
        self._step_count = 0
        self._simt = 0.0

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
        alt_m = alt * _FT_TO_M
        spd_mps = spd * _KTS_TO_MS
        # BlueSky's cre() interprets speed as CAS, so convert TAS → CAS
        cas_mps = vtas2cas(spd_mps, alt_m) if vtas2cas is not None else spd_mps
        bs.traf.cre(acid, actype, lat, lon, hdg, alt_m, cas_mps)
        self._managed_aircraft.add(acid)

    def remove_aircraft(self, acid: str) -> None:
        """Remove an aircraft.

        Args:
            acid: Aircraft ID to remove
        """
        bs.stack.stack(f"DELETE {acid}")
        self._managed_aircraft.discard(acid)

    def create_conflict_aircraft(
        self,
        ownship_lat: float,
        ownship_lon: float,
        ownship_alt: float,
        ownship_hdg: float,
        ownship_spd: float,
        count: int = 5,
        dpsi: float = 45.0,
        dcpa: float = 5.0,
        tlosh: float = 120.0,
        dH: float | None = None,
        tlosv: float | None = None,
        prefix: str = "CR",
        dpsi_list: list[float] | None = None,
        dcpa_list: list[float] | None = None,
        tlosh_list: list[float] | None = None,
    ) -> list[str]:
        """Create conflict aircraft using BlueSky's creconfs.

        Creates one ownship and ``count`` intruder aircraft using
        ``bs.traf.creconfs`` so that loss-of-separation events occur
        at a predictable time.

        Args:
            ownship_lat: Ownship latitude (deg).
            ownship_lon: Ownship longitude (deg).
            ownship_alt: Ownship altitude (ft).
            ownship_hdg: Ownship heading (deg).
            ownship_spd: Ownship speed (kts).
            count: Number of intruder aircraft to create.
            dpsi: Conflict angle spread (deg). Overridden by dpsi_list per intruder.
            dcpa: Distance at closest point of approach (NM). Overridden by dcpa_list per intruder.
            tlosh: Horizontal time to loss of separation (sec).
                Overridden by tlosh_list per intruder.
            dH: Vertical distance offset (ft). None = same altitude.
            tlosv: Vertical time to loss of separation (sec).
            prefix: Callsign prefix for intruder aircraft.
            dpsi_list: Per-intruder dpsi values. Overrides scalar dpsi.
            dcpa_list: Per-intruder dcpa values. Overrides scalar dcpa.
            tlosh_list: Per-intruder tlosh values. Overrides scalar tlosh.

        Returns:
            List of all created aircraft IDs (ownship + intruders).
        """
        # Create ownship
        own_id = f"{prefix}000"
        self.create_aircraft(
            own_id, "B737", ownship_lat, ownship_lon, ownship_alt, ownship_hdg, ownship_spd
        )
        own_idx = self._resolve_idx(own_id)

        # Create intruders via creconfs
        intruder_ids: list[str] = []
        for i in range(count):
            acid = f"{prefix}{i + 1:03d}"
            angle = float(dpsi_list[i] if dpsi_list is not None else dpsi * (i + 1) / max(count, 1))
            cpa = float(dcpa_list[i] if dcpa_list is not None else dcpa)
            tsh = float(tlosh_list[i] if tlosh_list is not None else tlosh)
            try:
                bs.traf.creconfs(
                    acid,
                    "B737",
                    own_idx,
                    angle,
                    cpa,
                    tsh,
                    dH=float(dH) if dH is not None else None,
                    tlosv=float(tlosv) if tlosv is not None else None,
                )
            except TypeError:
                # BlueSky creconfs may fail with numpy arrays in windfield
                # Fallback: create intruder at offset position
                from bluesky_pettingzoo.utils.geometry import point_at_distance

                offset_nm = max(cpa, 1.0)
                intr_lat, intr_lon = point_at_distance(
                    ownship_lat,
                    ownship_lon,
                    offset_nm,
                    ownship_hdg + angle,
                )
                intr_alt = ownship_alt + (float(dH) if dH is not None else 0.0)
                self.create_aircraft(
                    acid,
                    "B737",
                    intr_lat,
                    intr_lon,
                    intr_alt,
                    ownship_hdg + angle,
                    ownship_spd,
                )
            intruder_ids.append(acid)

        return [own_id] + intruder_ids

    def send_command(self, command: str) -> None:
        """Send a single command to BlueSky.

        Args:
            command: BlueSky command string
        """
        bs.stack.stack(command)

    def set_origin(self, acid: str, lat: float, lon: float) -> None:
        """Set aircraft origin waypoint.

        Args:
            acid: Aircraft ID
            lat: Origin latitude (degrees)
            lon: Origin longitude (degrees)
        """
        bs.stack.stack(f"ORIG {acid} {lat} {lon}")

    def set_destination(self, acid: str, lat: float, lon: float) -> None:
        """Set aircraft destination waypoint.

        Args:
            acid: Aircraft ID
            lat: Destination latitude (degrees)
            lon: Destination longitude (degrees)
        """
        bs.stack.stack(f"DEST {acid} {lat} {lon}")

    def add_waypoint(self, acid: str, lat: float, lon: float) -> None:
        """Add a waypoint to the aircraft's route.

        Args:
            acid: Aircraft ID
            lat: Waypoint latitude (degrees)
            lon: Waypoint longitude (degrees)
        """
        bs.stack.stack(f"ADDWPT {acid} {lat} {lon}")

    def enable_lnav(self, acid: str) -> None:
        """Enable lateral navigation (LNAV) for an aircraft.

        Once enabled, the aircraft will automatically follow its route
        waypoints using BlueSky's built-in navigation system.

        Args:
            acid: Aircraft ID
        """
        bs.stack.stack(f"LNAV {acid} ON")

    def disable_lnav(self, acid: str) -> None:
        """Disable lateral navigation (LNAV) for an aircraft.

        Args:
            acid: Aircraft ID
        """
        bs.stack.stack(f"LNAV {acid} OFF")

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
            "alt": float(bs.traf.alt[idx]) / _FT_TO_M,
            "hdg": float(bs.traf.hdg[idx]),
            "tas": float(bs.traf.tas[idx]) / _KTS_TO_MS,
            "vs": float(bs.traf.vs[idx]) * 60.0 / _FT_TO_M,
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
                "alt": float(bs.traf.alt[i]) / _FT_TO_M,
                "hdg": float(bs.traf.hdg[i]),
                "tas": float(bs.traf.tas[i]) / _KTS_TO_MS,
                "vs": float(bs.traf.vs[i]) * 60.0 / _FT_TO_M,
            }
        return states

    def set_aircraft_state(self, acid: str, **kwargs: Any) -> None:
        """Set state fields of an aircraft.

        Args:
            acid: Aircraft ID.
            **kwargs: State fields to set (lat, lon, alt, hdg, tas, vs).
                alt is in feet, tas in knots, vs in ft/min.
        """
        idx = self._resolve_idx(acid)
        if idx < 0:
            raise ValueError(f"Aircraft {acid} not found")
        if "lat" in kwargs:
            bs.traf.lat[idx] = kwargs["lat"]
        if "lon" in kwargs:
            bs.traf.lon[idx] = kwargs["lon"]
        if "alt" in kwargs:
            bs.traf.alt[idx] = kwargs["alt"] * _FT_TO_M
        if "hdg" in kwargs:
            bs.traf.hdg[idx] = kwargs["hdg"]
        if "tas" in kwargs:
            bs.traf.tas[idx] = kwargs["tas"] * _KTS_TO_MS
        if "vs" in kwargs:
            bs.traf.vs[idx] = kwargs["vs"] * _FT_TO_M / 60.0

    def set_vertical_control(
        self,
        acid: str,
        vs_kts: float,
        target_alt_ft: float | None = None,
    ) -> None:
        """Set vertical control via selalt/selvs, bypassing ALT command stack.

        Follows the bluesky-gym pattern: disable VNAV, then set selalt and
        selvs directly.  For a climb ``target_alt_ft`` should be a large
        value (e.g. 1_000_000); for descent it should be 0.

        Args:
            acid: Aircraft ID.
            vs_kts: Vertical speed in ft/min (positive = climb).
            target_alt_ft: Target altitude in feet.  If ``None``, uses
                1_000_000 for climb (vs > 0) or 0 for descent (vs <= 0).
        """
        idx = self._resolve_idx(acid)
        if idx < 0:
            raise ValueError(f"Aircraft {acid} not found")

        # Disable VNAV so selalt/selvs take effect
        bs.traf.swvnav[idx] = False

        # Convert ft/min → m/s for selvs
        vs_ms = vs_kts * _FT_TO_M / 60.0
        bs.traf.selvs[idx] = vs_ms

        # Set target altitude
        if target_alt_ft is None:
            target_alt_ft = 1_000_000.0 if vs_kts > 0 else 0.0
        bs.traf.selalt[idx] = target_alt_ft * _FT_TO_M

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

    def set_performance_model(self, model: str) -> None:
        """Switch performance model at runtime.

        Sends ``PERF {model}`` to the BlueSky stack.  Valid values are
        ``"openap"``, ``"bada"``, and ``"off"``.

        Args:
            model: Performance model name (``"openap"``, ``"bada"``, or ``"off"``).

        Raises:
            RuntimeError: If simulation has not been initialized.
            ValueError: If model name is not recognized.
        """
        if not self._initialized:
            raise RuntimeError("Simulation not initialized. Call init_simulation() first.")
        valid = {"openap", "bada", "off"}
        if model.lower() not in valid:
            raise ValueError(f"Invalid performance model '{model}'. Must be one of: {valid}")
        bs.stack.stack(f"PERF {model}")

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
