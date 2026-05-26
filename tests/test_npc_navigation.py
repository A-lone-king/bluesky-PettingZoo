"""Tests for NPC navigation — background aircraft with real waypoints.

When a scenario supports NPC navigation, background aircraft should have
BlueSky LNAV waypoints and conflict resolution should be disabled.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import numpy as np
import pytest

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.utils.types import AircraftState, ConflictConfig, SpawnConfig


class TestBaseScenarioNpcDefault:
    """BaseScenario should have a default configure_npc_navigation that returns empty list."""

    def test_default_returns_empty_list(self):
        class _MinimalScenario(BaseScenario):
            name = "Minimal"
            _action_space_type = "discrete"
            def setup(self, rng, airspace_bounds): return ["AC000"]
            def get_spawn_config(self): return SpawnConfig(altitude_range=(30000, 40000), speed_range=(400, 500), heading_range=(0, 360))
            def get_conflict_config(self): return ConflictConfig(nmac_horizontal_nm=5.0, nmac_vertical_ft=1000.0, warning_horizontal_nm=10.0, warning_vertical_ft=2000.0)
            def get_waypoint(self, agent_id): return {"lat": 40.5, "lon": 117.5}

        scenario = _MinimalScenario()
        wrapper = MagicMock()
        result = scenario.configure_npc_navigation(wrapper)
        assert result == []
        wrapper.send_command.assert_not_called()


class TestMergeScenarioNPC:
    """MergeScenario should configure background aircraft navigation."""

    def test_background_aircraft_get_nav_commands(self):
        from bluesky_pettingzoo.envs.scenarios.merge import MergeScenario

        scenario = MergeScenario(num_aircraft=5)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)

        wrapper = MagicMock()
        commands = scenario.configure_npc_navigation(wrapper)

        # Should send commands for background aircraft (AC001-AC004)
        assert len(commands) > 0
        # Should send reso off
        wrapper.send_command.assert_any_call("reso off")

    def test_controllable_aircraft_excluded(self):
        from bluesky_pettingzoo.envs.scenarios.merge import MergeScenario

        scenario = MergeScenario(num_aircraft=5)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)

        wrapper = MagicMock()
        commands = scenario.configure_npc_navigation(wrapper)

        # No NAV command for controllable aircraft AC000
        nav_cmds_for_ctrl = [c for c in commands if "AC000" in c and "NAV" in c]
        assert len(nav_cmds_for_ctrl) == 0

    def test_reso_off_called(self):
        from bluesky_pettingzoo.envs.scenarios.merge import MergeScenario

        scenario = MergeScenario(num_aircraft=3)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)

        wrapper = MagicMock()
        scenario.configure_npc_navigation(wrapper)

        wrapper.send_command.assert_any_call("reso off")
