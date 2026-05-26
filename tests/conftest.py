"""Shared test fixtures for bluesky-marl."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


@pytest.fixture
def bluesky_wrapper(default_config: dict[str, Any]):
    """Provide a fresh BlueSkyWrapper with real BlueSky simulation.

    Initializes BlueSky on first call (session-wide idempotent),
    resets aircraft state before each test, and closes after.
    """
    from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper

    wrapper = BlueSkyWrapper(default_config)
    wrapper.init_simulation()
    wrapper.reset()
    yield wrapper
    wrapper.close()

CONFIG_DIR = Path(__file__).parent.parent / "config"


@pytest.fixture
def default_config() -> dict[str, Any]:
    """Load default configuration."""
    with open(CONFIG_DIR / "default.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def rewards_config() -> dict[str, Any]:
    """Load rewards configuration."""
    with open(CONFIG_DIR / "rewards.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_aircraft_state() -> AircraftState:
    """Create a sample aircraft state."""
    return AircraftState(
        id="AC001",
        lat=39.25,
        lon=116.25,
        alt=35000.0,
        hdg=90.0,
        tas=450.0,
        vs=0.0,
    )


@pytest.fixture
def sample_aircraft_states() -> dict[str, AircraftState]:
    """Create multiple sample aircraft states."""
    return {
        "AC001": AircraftState(
            id="AC001",
            lat=39.25,
            lon=116.25,
            alt=35000.0,
            hdg=90.0,
            tas=450.0,
            vs=0.0,
        ),
        "AC002": AircraftState(
            id="AC002",
            lat=39.30,
            lon=116.30,
            alt=34000.0,
            hdg=270.0,
            tas=440.0,
            vs=0.0,
        ),
        "AC003": AircraftState(
            id="AC003",
            lat=39.20,
            lon=116.20,
            alt=36000.0,
            hdg=45.0,
            tas=460.0,
            vs=500.0,
        ),
    }


@pytest.fixture
def sample_discrete_action() -> DiscreteAction:
    """Create a sample discrete action."""
    return DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)


@pytest.fixture
def heading_adjustments() -> list[int]:
    """Heading adjustment options."""
    return [-20, -10, 0, 10, 20]


@pytest.fixture
def altitude_adjustments() -> list[int]:
    """Altitude adjustment options."""
    return [-2000, -1000, 0, 1000, 2000]


@pytest.fixture
def speed_adjustments() -> list[int]:
    """Speed adjustment options."""
    return [-20, -10, 0, 10, 20]
