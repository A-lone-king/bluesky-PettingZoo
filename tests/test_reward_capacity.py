"""Tests for capacity violation penalty reward component (T-V14)."""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.rewards.components.capacity import CapacityPenalty
from tests.helpers.state_factory import make_action, make_state

# Shared sector definitions for per-sector tests
_SECTORS = [
    {"id": "sector_a", "bounds": [[39.0, 116.0], [39.5, 116.5]], "capacity": 3},
    {"id": "sector_b", "bounds": [[39.0, 116.5], [39.5, 117.0]], "capacity": 2},
]


@pytest.fixture()
def capacity_config() -> dict:
    """Minimal config with capacity penalty component."""
    return {
        "components": {
            "capacity": {
                "enabled": True,
                "weight": 1.0,
                "max_aircraft": 5,
                "penalty_per_excess": -10.0,
            },
        },
    }


class TestCapacityNoPenaltyUnderLimit:
    """No penalty when aircraft count is at or below the limit."""

    def test_capacity_no_penalty_under_limit(self, capacity_config: dict) -> None:
        """3 aircraft with limit 5 → no penalty."""
        comp = CapacityPenalty(capacity_config)
        own = make_state("OWN")
        all_states = {
            "OWN": own,
            "AC001": make_state("AC001", lat=39.30),
            "AC002": make_state("AC002", lat=39.35),
        }
        action = make_action()

        result = comp.compute("OWN", own, action, own, all_states)
        assert result == 0.0


class TestCapacityPenaltyOverLimit:
    """Penalty should be given when aircraft count exceeds the limit."""

    def test_capacity_penalty_over_limit(self, capacity_config: dict) -> None:
        """7 aircraft with limit 5 → penalty for 2 excess."""
        comp = CapacityPenalty(capacity_config)
        own = make_state("OWN")
        all_states = {"OWN": own}
        for i in range(6):
            all_states[f"AC{i:03d}"] = make_state(f"AC{i:03d}", lat=39.25 + i * 0.01)
        action = make_action()

        result = comp.compute("OWN", own, action, own, all_states)
        # 7 aircraft, limit 5, excess = 2, penalty = 2 * -10 = -20
        assert result == -20.0


class TestCapacityPenaltyProportional:
    """Penalty should be proportional to the number of excess aircraft."""

    def test_capacity_penalty_proportional(self, capacity_config: dict) -> None:
        """More excess aircraft → more negative penalty."""
        comp = CapacityPenalty(capacity_config)
        action = make_action()

        # 8 aircraft (excess 3)
        states_8 = {f"AC{i:03d}": make_state(f"AC{i:03d}") for i in range(8)}
        result_8 = comp.compute("AC000", states_8["AC000"], action, states_8["AC000"], states_8)

        # 10 aircraft (excess 5)
        states_10 = {f"AC{i:03d}": make_state(f"AC{i:03d}") for i in range(10)}
        result_10 = comp.compute("AC000", states_10["AC000"], action, states_10["AC000"], states_10)

        assert result_8 < 0
        assert result_10 < result_8  # More excess → more negative


class TestCapacityThresholdConfigurable:
    """Capacity threshold should be configurable."""

    def test_capacity_threshold_configurable(self, capacity_config: dict) -> None:
        """Changing max_aircraft changes when penalty kicks in."""
        # With max_aircraft=3, 4 aircraft should trigger penalty
        capacity_config["components"]["capacity"]["max_aircraft"] = 3
        comp = CapacityPenalty(capacity_config)
        own = make_state("OWN")
        all_states = {
            "OWN": own,
            "AC001": make_state("AC001"),
            "AC002": make_state("AC002"),
            "AC003": make_state("AC003"),
        }
        action = make_action()

        result = comp.compute("OWN", own, action, own, all_states)
        # 4 aircraft, limit 3, excess = 1
        assert result == -10.0


class TestCapacityPenaltyReset:
    """Reset should not crash (stateless component)."""

    def test_capacity_penalty_reset(self, capacity_config: dict) -> None:
        """reset() completes without error and compute still works."""
        comp = CapacityPenalty(capacity_config)
        comp.reset()

        own = make_state("OWN")
        all_states = {"OWN": own}
        action = make_action()
        result = comp.compute("OWN", own, action, own, all_states)
        assert result == 0.0


# ---------------------------------------------------------------------------
# Per-sector capacity tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def sector_config() -> dict:
    """Config with per-sector capacity definitions."""
    return {
        "components": {
            "capacity": {
                "enabled": True,
                "weight": 1.0,
                "sectors": _SECTORS,
                "warning_threshold": 0.8,
                "penalty_per_excess": -10.0,
            },
        },
    }


class TestSectorCapacityNoPenalty:
    """No penalty when each sector is under capacity."""

    def test_no_penalty_under_capacity(self, sector_config: dict) -> None:
        """2 aircraft in sector_a (cap 3) and 1 in sector_b (cap 2) → no penalty."""
        comp = CapacityPenalty(sector_config)
        action = make_action()
        all_states = {
            "AC000": make_state("AC000", lat=39.2, lon=116.2),  # sector_a
            "AC001": make_state("AC001", lat=39.3, lon=116.3),  # sector_a
            "AC002": make_state("AC002", lat=39.2, lon=116.7),  # sector_b
        }
        result = comp.compute("AC000", all_states["AC000"], action, all_states["AC000"], all_states)
        assert result == 0.0


class TestSectorCapacityPenaltyExcess:
    """Penalty when a sector exceeds its capacity."""

    def test_penalty_one_sector_over(self, sector_config: dict) -> None:
        """4 aircraft in sector_a (cap 3) → penalty for 1 excess."""
        comp = CapacityPenalty(sector_config)
        action = make_action()
        all_states = {
            f"AC{i:03d}": make_state(f"AC{i:03d}", lat=39.1 + i * 0.05, lon=116.2) for i in range(4)
        }
        result = comp.compute("AC000", all_states["AC000"], action, all_states["AC000"], all_states)
        # 1 excess * -10 = -10
        assert result == -10.0

    def test_penalty_both_sectors_over(self, sector_config: dict) -> None:
        """Penalty sums across sectors when both exceed capacity."""
        comp = CapacityPenalty(sector_config)
        action = make_action()
        all_states = {
            # 4 in sector_a (cap 3, excess 1)
            "AC000": make_state("AC000", lat=39.1, lon=116.2),
            "AC001": make_state("AC001", lat=39.2, lon=116.2),
            "AC002": make_state("AC002", lat=39.3, lon=116.2),
            "AC003": make_state("AC003", lat=39.4, lon=116.2),
            # 3 in sector_b (cap 2, excess 1)
            "AC004": make_state("AC004", lat=39.1, lon=116.7),
            "AC005": make_state("AC005", lat=39.2, lon=116.7),
            "AC006": make_state("AC006", lat=39.3, lon=116.7),
        }
        # Agent in sector_a: sector_a excess=1 → -10
        result = comp.compute("AC000", all_states["AC000"], action, all_states["AC000"], all_states)
        assert result == -10.0

    def test_penalty_ignores_other_sector_for_agent(self, sector_config: dict) -> None:
        """Agent only penalized for its own sector's excess."""
        comp = CapacityPenalty(sector_config)
        action = make_action()
        all_states = {
            # 1 in sector_a (under cap)
            "AC000": make_state("AC000", lat=39.2, lon=116.2),
            # 3 in sector_b (cap 2, excess 1)
            "AC001": make_state("AC001", lat=39.1, lon=116.7),
            "AC002": make_state("AC002", lat=39.2, lon=116.7),
            "AC003": make_state("AC003", lat=39.3, lon=116.7),
        }
        # Agent AC000 is in sector_a which is under capacity → 0
        result = comp.compute("AC000", all_states["AC000"], action, all_states["AC000"], all_states)
        assert result == 0.0


class TestSectorCapacityAgentOutside:
    """Agent not in any sector should get no penalty."""

    def test_agent_outside_sectors(self, sector_config: dict) -> None:
        """Agent outside all sectors gets 0 penalty regardless of global count."""
        comp = CapacityPenalty(sector_config)
        action = make_action()
        all_states = {
            "AC000": make_state("AC000", lat=40.0, lon=116.2),  # outside
            "AC001": make_state("AC001", lat=39.2, lon=116.2),  # sector_a
            "AC002": make_state("AC002", lat=39.3, lon=116.2),  # sector_a
        }
        result = comp.compute("AC000", all_states["AC000"], action, all_states["AC000"], all_states)
        assert result == 0.0


class TestSectorCapacityWarning:
    """Warning threshold should produce intermediate penalty."""

    def test_warning_threshold(self, sector_config: dict) -> None:
        """Aircraft count at warning threshold gets warning penalty."""
        comp = CapacityPenalty(sector_config)
        action = make_action()
        # 3 aircraft in sector_a (cap 3, warning at 0.8 → warning at ceil(3*0.8)=3)
        # At exactly capacity, should be at warning level (not excess)
        all_states = {
            "AC000": make_state("AC000", lat=39.1, lon=116.2),
            "AC001": make_state("AC001", lat=39.2, lon=116.2),
            "AC002": make_state("AC002", lat=39.3, lon=116.2),
        }
        result = comp.compute("AC000", all_states["AC000"], action, all_states["AC000"], all_states)
        # At capacity (3/3), warning_threshold=0.8 → count >= ceil(3*0.8)=3 → warning
        assert result < 0  # warning penalty applied


class TestSectorCapacityReset:
    """Reset should not crash (stateless component)."""

    def test_reset_no_crash(self, sector_config: dict) -> None:
        """reset() completes without error."""
        comp = CapacityPenalty(sector_config)
        comp.reset()
        action = make_action()
        own = make_state("AC000")
        result = comp.compute("AC000", own, action, own, {"AC000": own})
        assert result == 0.0
