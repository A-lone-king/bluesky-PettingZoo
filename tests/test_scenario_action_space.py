"""Tests for scenario action space interface (spec4 F2).

Verify action_space_type and continuous_action_dims properties.
"""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario


class TestActionSpaceType:
    """BaseScenario should have action_space_type property."""

    def test_default_is_discrete(self) -> None:
        """Default action_space_type should be 'discrete'."""

        class DummyScenario(BaseScenario):
            def setup(self, rng, airspace_bounds):
                return []

            def get_spawn_config(self):
                from bluesky_pettingzoo.utils.types import SpawnConfig
                return SpawnConfig(
                    altitude_range=(30000, 40000),
                    speed_range=(400, 500),
                    heading_range=(0, 360),
                )

            def get_conflict_config(self):
                from bluesky_pettingzoo.utils.types import ConflictConfig
                return ConflictConfig(
                    nmac_horizontal_nm=5,
                    nmac_vertical_ft=1000,
                    warning_horizontal_nm=10,
                    warning_vertical_ft=2000,
                )

            def get_waypoint(self, agent_id):
                return {"lat": 39.5, "lon": 116.5, "alt": 35000, "hdg": 90}

        scenario = DummyScenario()
        assert scenario.action_space_type == "discrete"

    def test_continuous_type(self) -> None:
        """Scenario with continuous action space should return 'continuous'."""

        class ContinuousScenario(BaseScenario):
            action_space_type = "continuous"

            def setup(self, rng, airspace_bounds):
                return []

            def get_spawn_config(self):
                from bluesky_pettingzoo.utils.types import SpawnConfig
                return SpawnConfig(
                    altitude_range=(30000, 40000),
                    speed_range=(400, 500),
                    heading_range=(0, 360),
                )

            def get_conflict_config(self):
                from bluesky_pettingzoo.utils.types import ConflictConfig
                return ConflictConfig(
                    nmac_horizontal_nm=5,
                    nmac_vertical_ft=1000,
                    warning_horizontal_nm=10,
                    warning_vertical_ft=2000,
                )

            def get_waypoint(self, agent_id):
                return {"lat": 39.5, "lon": 116.5, "alt": 35000, "hdg": 90}

        scenario = ContinuousScenario()
        assert scenario.action_space_type == "continuous"


class TestContinuousActionDims:
    """BaseScenario should have continuous_action_dims property."""

    def test_default_dims(self) -> None:
        """Default continuous_action_dims should be 3 (heading, altitude, speed)."""

        class DummyScenario(BaseScenario):
            def setup(self, rng, airspace_bounds):
                return []

            def get_spawn_config(self):
                from bluesky_pettingzoo.utils.types import SpawnConfig
                return SpawnConfig(
                    altitude_range=(30000, 40000),
                    speed_range=(400, 500),
                    heading_range=(0, 360),
                )

            def get_conflict_config(self):
                from bluesky_pettingzoo.utils.types import ConflictConfig
                return ConflictConfig(
                    nmac_horizontal_nm=5,
                    nmac_vertical_ft=1000,
                    warning_horizontal_nm=10,
                    warning_vertical_ft=2000,
                )

            def get_waypoint(self, agent_id):
                return {"lat": 39.5, "lon": 116.5, "alt": 35000, "hdg": 90}

        scenario = DummyScenario()
        assert scenario.continuous_action_dims == 3
