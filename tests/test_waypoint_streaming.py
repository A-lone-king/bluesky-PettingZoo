"""Tests for waypoint streaming — arriving at a waypoint generates a new one.

When a scenario supports waypoint streaming, reaching the current waypoint
should NOT terminate the agent. Instead, the scenario provides a new waypoint
and the agent continues flying.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from gymnasium import spaces

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.types import AircraftState, ConflictConfig, SpawnConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(acid="AC000", lat=40.0, lon=117.0, alt=35000.0, hdg=90.0, tas=450.0, vs=0.0):
    return AircraftState(
        id=acid, lat=lat, lon=lon, alt=alt, hdg=hdg, tas=tas, vs=vs,
    )


class _StreamingScenario(BaseScenario):
    """Scenario that supports waypoint streaming."""

    name = "StreamingTest"
    _action_space_type = "discrete"
    _waypoint_sequence = [
        {"lat": 40.5, "lon": 117.5},
        {"lat": 41.0, "lon": 118.0},
        {"lat": 41.5, "lon": 118.5},
    ]
    _wp_index: dict[str, int] = {}

    def setup(self, rng, airspace_bounds):
        self._wp_index = {"AC000": 0}
        return ["AC000"]

    def get_spawn_config(self):
        return SpawnConfig(
            altitude_range=(30000, 40000),
            speed_range=(400, 500),
            heading_range=(0, 360),
        )

    def get_conflict_config(self):
        return ConflictConfig(
            nmac_horizontal_nm=5.0,
            nmac_vertical_ft=1000.0,
            warning_horizontal_nm=10.0,
            warning_vertical_ft=2000.0,
        )

    def get_waypoint(self, agent_id: str):
        idx = self._wp_index.get(agent_id, 0)
        return self._waypoint_sequence[idx]

    def update_waypoint(self, agent_id: str, state: AircraftState) -> dict[str, float] | None:
        """Return next waypoint if available, None if done."""
        idx = self._wp_index.get(agent_id, 0) + 1
        if idx < len(self._waypoint_sequence):
            self._wp_index[agent_id] = idx
            return self._waypoint_sequence[idx]
        return None


# ---------------------------------------------------------------------------
# Tests: BaseScenario default
# ---------------------------------------------------------------------------

class TestBaseScenarioDefault:
    """BaseScenario.update_waypoint() should return None by default."""

    def test_default_update_waypoint_returns_none(self):
        """Default implementation should return None (no streaming)."""
        scenario = _StreamingScenario()
        state = _make_state()
        # Even though _StreamingScenario overrides update_waypoint,
        # BaseScenario's default should be None
        # Use a minimal scenario that doesn't override
        class _MinimalScenario(BaseScenario):
            name = "Minimal"
            _action_space_type = "discrete"
            def setup(self, rng, airspace_bounds): return ["AC000"]
            def get_spawn_config(self): return SpawnConfig(altitude_range=(30000, 40000), speed_range=(400, 500), heading_range=(0, 360))
            def get_conflict_config(self): return ConflictConfig(nmac_horizontal_nm=5.0, nmac_vertical_ft=1000.0, warning_horizontal_nm=10.0, warning_vertical_ft=2000.0)
            def get_waypoint(self, agent_id): return {"lat": 40.5, "lon": 117.5}

        minimal = _MinimalScenario()
        result = minimal.update_waypoint("AC000", state)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Streaming scenario
# ---------------------------------------------------------------------------

class TestStreamingScenario:
    """Streaming scenario should provide sequential waypoints."""

    def test_first_update_returns_second_waypoint(self):
        scenario = _StreamingScenario()
        scenario.setup(np.random.RandomState(42), {})
        state = _make_state()
        result = scenario.update_waypoint("AC000", state)
        assert result == {"lat": 41.0, "lon": 118.0}

    def test_second_update_returns_third_waypoint(self):
        scenario = _StreamingScenario()
        scenario.setup(np.random.RandomState(42), {})
        state = _make_state()
        scenario.update_waypoint("AC000", state)
        result = scenario.update_waypoint("AC000", state)
        assert result == {"lat": 41.5, "lon": 118.5}

    def test_exhausted_waypoints_returns_none(self):
        scenario = _StreamingScenario()
        scenario.setup(np.random.RandomState(42), {})
        state = _make_state()
        scenario.update_waypoint("AC000", state)
        scenario.update_waypoint("AC000", state)
        result = scenario.update_waypoint("AC000", state)
        assert result is None
