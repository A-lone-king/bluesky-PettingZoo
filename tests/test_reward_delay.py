"""Tests for delay penalty reward component (T-V13)."""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.rewards.components.delay import DelayPenalty
from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction


def make_state(
    acid: str = "OWN",
    lat: float = 39.25,
    lon: float = 116.25,
    alt: float = 35000.0,
    hdg: float = 90.0,
    tas: float = 450.0,
    vs: float = 0.0,
) -> AircraftState:
    return AircraftState(
        id=acid, lat=lat, lon=lon, alt=alt, hdg=hdg, tas=tas, vs=vs,
    )


def make_action() -> DiscreteAction:
    return DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)


@pytest.fixture()
def delay_config() -> dict:
    """Minimal config with delay penalty component."""
    return {
        "components": {
            "delay": {
                "enabled": True,
                "weight": 1.0,
                "penalty": -0.01,
            },
        },
    }


class TestDelayPenaltyPerStep:
    """Each step should return a fixed penalty."""

    def test_delay_penalty_per_step(self, delay_config: dict) -> None:
        """Default penalty is -0.01 per step."""
        comp = DelayPenalty(delay_config)
        own = make_state()
        action = make_action()
        all_states = {"OWN": own}

        result = comp.compute("OWN", own, action, own, all_states)
        assert result == -0.01


class TestDelayPenaltyConfigurable:
    """Penalty value should be configurable."""

    def test_delay_penalty_configurable(self, delay_config: dict) -> None:
        """Custom penalty value is used."""
        delay_config["components"]["delay"]["penalty"] = -0.05
        comp = DelayPenalty(delay_config)
        own = make_state()
        action = make_action()
        all_states = {"OWN": own}

        result = comp.compute("OWN", own, action, own, all_states)
        assert result == -0.05


class TestDelayPenaltyWeight:
    """Weight should correctly scale the penalty."""

    def test_delay_penalty_weight(self, delay_config: dict) -> None:
        """Weight parameter is accessible from config."""
        delay_config["components"]["delay"]["weight"] = 2.0
        comp = DelayPenalty(delay_config)
        # Weight is applied by RewardCalculator, not by the component itself.
        # The component just returns the raw penalty.
        own = make_state()
        action = make_action()
        all_states = {"OWN": own}

        result = comp.compute("OWN", own, action, own, all_states)
        # Raw penalty is still -0.01; weight is handled externally
        assert result == -0.01


class TestDelayPenaltyReset:
    """Reset should not crash (stateless component)."""

    def test_delay_penalty_reset(self, delay_config: dict) -> None:
        """reset() completes without error."""
        comp = DelayPenalty(delay_config)
        comp.reset()
        # After reset, compute should still work normally
        own = make_state()
        action = make_action()
        result = comp.compute("OWN", own, action, own, {"OWN": own})
        assert result == -0.01
