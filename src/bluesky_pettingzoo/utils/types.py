"""Type definitions for bluesky-marl environment."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, NamedTuple, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

from bluesky_pettingzoo.utils.mixin import DictBackedMixin

# Agent identifier
AgentID: TypeAlias = str


class AircraftState(DictBackedMixin):
    """BlueSky aircraft state.

    Supports both attribute access (state.id) and dict access (state["id"]).
    """

    __slots__ = ("id", "lat", "lon", "alt", "hdg", "tas", "vs")

    def __init__(
        self,
        *,
        id: str,
        lat: float,
        lon: float,
        alt: float,
        hdg: float,
        tas: float,
        vs: float,
    ) -> None:
        self.id = id
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.hdg = hdg
        self.tas = tas
        self.vs = vs

    def __repr__(self) -> str:
        return (
            f"AircraftState(id={self.id!r}, lat={self.lat}, lon={self.lon}, "
            f"alt={self.alt}, hdg={self.hdg}, tas={self.tas}, vs={self.vs})"
        )


class NormalizedObservation(TypedDict):
    """Normalized observation data."""

    self_state: NDArray[np.float32]  # shape=(6,)
    other_aircraft: NDArray[np.float32]  # shape=(MAX_OBS, 7)
    other_aircraft_mask: NDArray[np.int8]  # shape=(MAX_OBS,)
    goal: NDArray[np.float32]  # shape=(4,)


class TextualState(TypedDict):
    """Textual state for LLM/RAG integration."""

    agent_id: str
    position: dict[str, float]
    heading: float
    altitude: float
    speed: float
    observable_aircraft: list[dict[str, Any]]
    conflict_status: str  # "safe" | "warning" | "nmac"
    text: str


class AirspaceSnapshot(TypedDict):
    """Airspace topology snapshot."""

    sectors: list[dict[str, Any]]
    waypoints: list[dict[str, Any]]
    aircraft_positions: dict[str, dict[str, float]]


class Route:
    """Ordered sequence of waypoints forming a route."""

    __slots__ = ("waypoints",)

    def __init__(self, waypoints: list[dict[str, float]]) -> None:
        self.waypoints = waypoints

    def total_distance_nm(self) -> float:
        """Total route distance in nautical miles."""
        from bluesky_pettingzoo.utils.geometry import haversine_distance

        total = 0.0
        for i in range(len(self.waypoints) - 1):
            w1, w2 = self.waypoints[i], self.waypoints[i + 1]
            total += haversine_distance(w1["lat"], w1["lon"], w2["lat"], w2["lon"])
        return total

    def segment_count(self) -> int:
        """Number of segments (waypoints - 1)."""
        return max(0, len(self.waypoints) - 1)

    def get_segment(self, index: int) -> tuple[tuple[float, float], tuple[float, float]]:
        """Get segment endpoints by index.

        Args:
            index: Segment index (0-based).

        Returns:
            ((lat1, lon1), (lat2, lon2)) for the segment.

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= self.segment_count():
            raise IndexError(f"Segment index {index} out of range (0..{self.segment_count() - 1})")
        w1, w2 = self.waypoints[index], self.waypoints[index + 1]
        return (w1["lat"], w1["lon"]), (w2["lat"], w2["lon"])


class DiscreteAction(NamedTuple):
    """Discrete action indices."""

    heading_idx: int  # 0-4
    altitude_idx: int  # 0-4
    speed_idx: int  # 0-4


class ContinuousAction(NamedTuple):
    """Continuous action values for BlueSky commands."""

    heading: float  # 0-360
    altitude: float  # feet
    speed: float  # knots


class ConflictLevel(IntEnum):
    """Conflict severity levels."""

    SAFE = 0
    WARNING = 1
    NMAC = 2


class SimulationConfig(DictBackedMixin):
    """Simulation parameters."""

    __slots__ = ("dt", "max_episode_steps", "headless", "action_frequency")

    def __init__(
        self,
        *,
        dt: float,
        max_episode_steps: int,
        headless: bool = True,
        action_frequency: int = 1,
    ) -> None:
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.headless = headless
        self.action_frequency = action_frequency


class SectorConfig(DictBackedMixin):
    """Sector definition."""

    __slots__ = ("id", "bounds")

    def __init__(self, *, id: str, bounds: list[list[float]]) -> None:
        self.id = id
        self.bounds = bounds


class WaypointConfig(DictBackedMixin):
    """Waypoint definition."""

    __slots__ = ("id", "lat", "lon", "alt")

    def __init__(self, *, id: str, lat: float, lon: float, alt: float) -> None:
        self.id = id
        self.lat = lat
        self.lon = lon
        self.alt = alt


class AirspaceConfig(DictBackedMixin):
    """Airspace definition with sectors and optional waypoints."""

    __slots__ = ("name", "sectors", "waypoints")

    def __init__(
        self,
        *,
        name: str,
        sectors: list[SectorConfig],
        waypoints: list[WaypointConfig] | None = None,
    ) -> None:
        self.name = name
        self.sectors = sectors
        self.waypoints = waypoints or []


class SpawnConfig(DictBackedMixin):
    """Aircraft spawn parameters."""

    __slots__ = ("altitude_range", "speed_range", "heading_range")

    def __init__(
        self,
        *,
        altitude_range: tuple[float, float],
        speed_range: tuple[float, float],
        heading_range: tuple[float, float],
    ) -> None:
        self.altitude_range = altitude_range
        self.speed_range = speed_range
        self.heading_range = heading_range

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpawnConfig):
            return NotImplemented
        return (
            self.altitude_range == other.altitude_range
            and self.speed_range == other.speed_range
            and self.heading_range == other.heading_range
        )


class DynamicEntryConfig(DictBackedMixin):
    """Dynamic aircraft entry parameters."""

    __slots__ = ("enabled", "interval", "max_total")

    def __init__(self, *, enabled: bool, interval: int, max_total: int) -> None:
        self.enabled = enabled
        self.interval = interval
        self.max_total = max_total


class ConflictConfig(DictBackedMixin):
    """Conflict detection thresholds."""

    __slots__ = (
        "nmac_horizontal_nm",
        "nmac_vertical_ft",
        "warning_horizontal_nm",
        "warning_vertical_ft",
    )

    def __init__(
        self,
        *,
        nmac_horizontal_nm: float,
        nmac_vertical_ft: float,
        warning_horizontal_nm: float,
        warning_vertical_ft: float,
    ) -> None:
        self.nmac_horizontal_nm = nmac_horizontal_nm
        self.nmac_vertical_ft = nmac_vertical_ft
        self.warning_horizontal_nm = warning_horizontal_nm
        self.warning_vertical_ft = warning_vertical_ft


class AircraftConfig(DictBackedMixin):
    """Aircraft configuration with spawn parameters."""

    __slots__ = ("initial_count", "spawn")

    def __init__(self, *, initial_count: int, spawn: SpawnConfig) -> None:
        self.initial_count = initial_count
        self.spawn = spawn


class ScenarioConfig(DictBackedMixin):
    """Top-level scenario configuration."""

    __slots__ = ("name", "simulation", "airspace", "aircraft", "dynamic_entry", "conflict")

    def __init__(
        self,
        *,
        name: str,
        simulation: SimulationConfig,
        airspace: AirspaceConfig,
        aircraft: AircraftConfig,
        dynamic_entry: DynamicEntryConfig | None = None,
        conflict: ConflictConfig | None = None,
    ) -> None:
        self.name = name
        self.simulation = simulation
        self.airspace = airspace
        self.aircraft = aircraft
        self.dynamic_entry = dynamic_entry
        self.conflict = conflict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScenarioConfig:
        """Create ScenarioConfig from a dictionary (e.g. parsed YAML)."""
        sim_data = data["simulation"]
        simulation = SimulationConfig(
            dt=sim_data["dt"],
            max_episode_steps=sim_data["max_episode_steps"],
            headless=sim_data.get("headless", True),
            action_frequency=sim_data.get("action_frequency", 1),
        )

        air_data = data["airspace"]
        sectors = [
            SectorConfig(id=s["id"], bounds=s["bounds"])
            for s in air_data["sectors"]
        ]
        waypoints = [
            WaypointConfig(id=w["id"], lat=w["lat"], lon=w["lon"], alt=w["alt"])
            for w in air_data.get("waypoints", [])
        ]
        airspace = AirspaceConfig(
            name=air_data["name"],
            sectors=sectors,
            waypoints=waypoints,
        )

        ac_data = data["aircraft"]
        spawn_data = ac_data["spawn"]
        spawn = SpawnConfig(
            altitude_range=tuple(spawn_data["altitude_range"]),
            speed_range=tuple(spawn_data["speed_range"]),
            heading_range=tuple(spawn_data["heading_range"]),
        )
        aircraft = AircraftConfig(
            initial_count=ac_data["initial_count"],
            spawn=spawn,
        )

        dynamic_entry = None
        if "dynamic_entry" in data:
            de = data["dynamic_entry"]
            dynamic_entry = DynamicEntryConfig(
                enabled=de["enabled"],
                interval=de["interval"],
                max_total=de["max_total"],
            )

        conflict = None
        if "conflict" in data:
            cf = data["conflict"]
            conflict = ConflictConfig(
                nmac_horizontal_nm=cf["nmac_horizontal_nm"],
                nmac_vertical_ft=cf["nmac_vertical_ft"],
                warning_horizontal_nm=cf["warning_horizontal_nm"],
                warning_vertical_ft=cf["warning_vertical_ft"],
            )

        return cls(
            name=data["name"],
            simulation=simulation,
            airspace=airspace,
            aircraft=aircraft,
            dynamic_entry=dynamic_entry,
            conflict=conflict,
        )
