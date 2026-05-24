"""Scenario definitions."""

from bluesky_pettingzoo.envs.scenarios.base import BaseScenario
from bluesky_pettingzoo.envs.scenarios.descent import DescentScenario
from bluesky_pettingzoo.envs.scenarios.horizontal_cr import HorizontalCRScenario
from bluesky_pettingzoo.envs.scenarios.merge import MergeScenario
from bluesky_pettingzoo.envs.scenarios.route_nav import RouteNavScenario
from bluesky_pettingzoo.envs.scenarios.sector_capacity import SectorCapacityScenario
from bluesky_pettingzoo.envs.scenarios.sector_cr import SectorCRScenario
from bluesky_pettingzoo.envs.scenarios.static_obstacle import StaticObstacleScenario
from bluesky_pettingzoo.envs.scenarios.vertical_cr import VerticalCRScenario
from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario

__all__ = [
    "BaseScenario",
    "DescentScenario",
    "HorizontalCRScenario",
    "MergeScenario",
    "RouteNavScenario",
    "SectorCapacityScenario",
    "SectorCRScenario",
    "StaticObstacleScenario",
    "VerticalCRScenario",
    "WaypointNavScenario",
]
