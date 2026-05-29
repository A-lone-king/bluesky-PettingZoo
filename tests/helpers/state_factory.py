"""Shared factory functions for creating test states and actions."""

from __future__ import annotations

from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


def make_state(
    acid: str = "AC001",
    lat: float = 39.25,
    lon: float = 116.25,
    alt: float = 35000.0,
    hdg: float = 90.0,
    tas: float = 450.0,
    vs: float = 0.0,
) -> AircraftState:
    """Create an AircraftState with default values.

    Args:
        acid: Aircraft identifier
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude in feet
        hdg: Heading in degrees
        tas: True airspeed in knots
        vs: Vertical speed in ft/min

    Returns:
        AircraftState instance
    """
    return AircraftState(
        id=acid, lat=lat, lon=lon, alt=alt, hdg=hdg, tas=tas, vs=vs,
    )


def make_action(
    heading_idx: int = 2,
    altitude_idx: int = 2,
    speed_idx: int = 2,
) -> DiscreteAction:
    """Create a DiscreteAction with default (noop) values.

    Args:
        heading_idx: Heading action index (2 = no change)
        altitude_idx: Altitude action index (2 = no change)
        speed_idx: Speed action index (2 = no change)

    Returns:
        DiscreteAction instance
    """
    return DiscreteAction(heading_idx=heading_idx, altitude_idx=altitude_idx, speed_idx=speed_idx)
