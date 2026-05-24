"""Tests for StaticObstacleScenario."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_pettingzoo.envs.scenarios.static_obstacle import (
    StaticObstacleScenario,
    _generate_obstacle_polygon,
    _polygon_area_nm2,
)
from bluesky_pettingzoo.utils.geometry import point_in_polygon


class TestStaticObstacleScenarioSetup:
    """Test scenario initialization and setup."""

    def test_setup_returns_agents(self) -> None:
        scenario = StaticObstacleScenario(num_aircraft=2, num_obstacles=5)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        agents = scenario.setup(rng, bounds)
        assert len(agents) == 2
        assert agents == ["AC000", "AC001"]

    def test_setup_generates_obstacles(self) -> None:
        scenario = StaticObstacleScenario(num_aircraft=1, num_obstacles=10)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)
        obstacles = scenario.get_obstacles()
        assert len(obstacles) == 10
        for polygon in obstacles:
            assert len(polygon) >= 3  # at least 3 vertices

    def test_setup_generates_obstacle_names(self) -> None:
        scenario = StaticObstacleScenario(num_aircraft=1, num_obstacles=5)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)
        names = scenario.get_obstacle_names()
        assert len(names) == 5
        assert names[0] == "restricted_area_1"
        assert names[4] == "restricted_area_5"

    def test_setup_provides_initial_positions(self) -> None:
        scenario = StaticObstacleScenario(num_aircraft=3)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)
        positions = scenario.get_initial_positions()
        assert positions is not None
        assert len(positions) == 3
        for acid, (lat, lon) in positions.items():
            assert 39.0 <= lat <= 41.0
            assert 116.0 <= lon <= 118.0

    def test_setup_assigns_waypoints(self) -> None:
        scenario = StaticObstacleScenario(num_aircraft=2, num_obstacles=5)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        agents = scenario.setup(rng, bounds)
        for acid in agents:
            wp = scenario.get_waypoint(acid)
            assert "lat" in wp
            assert "lon" in wp
            assert "alt" in wp
            assert "hdg" in wp

    def test_waypoints_outside_obstacles(self) -> None:
        """Waypoints should not be inside any obstacle polygon."""
        scenario = StaticObstacleScenario(num_aircraft=2, num_obstacles=10)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        agents = scenario.setup(rng, bounds)
        obstacles = scenario.get_obstacles()
        for acid in agents:
            wp = scenario.get_waypoint(acid)
            for polygon in obstacles:
                assert not point_in_polygon(wp["lat"], wp["lon"], polygon), (
                    f"{acid} waypoint ({wp['lat']}, {wp['lon']}) is inside an obstacle"
                )


class TestStaticObstacleScenarioConfig:
    """Test scenario configuration methods."""

    def test_action_dimensions(self) -> None:
        scenario = StaticObstacleScenario()
        assert scenario.action_dimensions == [0, 2]  # heading + speed

    def test_spawn_config(self) -> None:
        scenario = StaticObstacleScenario()
        spawn = scenario.get_spawn_config()
        assert spawn.altitude_range == (35000.0, 35000.0)
        assert spawn.speed_range == (150.0, 150.0)
        assert spawn.heading_range == (0, 360)

    def test_conflict_config(self) -> None:
        scenario = StaticObstacleScenario()
        conf = scenario.get_conflict_config()
        assert conf.nmac_horizontal_nm == 5.0
        assert conf.nmac_vertical_ft == 1000.0

    def test_should_truncate_always_false(self) -> None:
        """Obstacle scenario never truncates — intrusion is termination."""
        from bluesky_pettingzoo.utils.types import AircraftState

        scenario = StaticObstacleScenario()
        state = AircraftState(id="AC000", lat=100.0, lon=200.0, alt=35000, hdg=90, tas=150, vs=0)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        assert scenario.should_truncate("AC000", state, bounds) is False

    def test_reset_clears_state(self) -> None:
        scenario = StaticObstacleScenario(num_aircraft=2, num_obstacles=5)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)
        scenario.reset()
        assert scenario.get_obstacles() == []
        assert scenario.get_initial_positions() is None


class TestGenerateObstaclePolygon:
    """Test polygon generation utility."""

    def test_polygon_has_minimum_vertices(self) -> None:
        rng = np.random.RandomState(42)
        polygon = _generate_obstacle_polygon(rng, 39.5, 116.5)
        assert len(polygon) >= 3

    def test_polygon_vertices_are_tuples(self) -> None:
        rng = np.random.RandomState(42)
        polygon = _generate_obstacle_polygon(rng, 39.5, 116.5)
        for v in polygon:
            assert isinstance(v, tuple)
            assert len(v) == 2

    def test_polygon_area_above_minimum(self) -> None:
        rng = np.random.RandomState(42)
        polygon = _generate_obstacle_polygon(rng, 39.5, 116.5)
        area = _polygon_area_nm2(polygon, 39.5)
        assert area >= 20  # generous lower bound (approximation may be rough)


class TestObstacleIntrusionComponent:
    """Test ObstacleIntrusion reward component."""

    def test_no_intrusion_returns_zero(self) -> None:
        from bluesky_pettingzoo.rewards.components.obstacle_intrusion import ObstacleIntrusion
        from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction

        comp = ObstacleIntrusion()
        comp.set_obstacles([[(40.0, 117.0), (40.1, 117.0), (40.05, 117.1)]])
        state = AircraftState(id="AC000", lat=39.0, lon=116.0, alt=35000, hdg=90, tas=150, vs=0)
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)
        reward = comp.compute("AC000", state, action, state, {"AC000": state})
        assert reward == 0.0

    def test_intrusion_returns_penalty(self) -> None:
        from bluesky_pettingzoo.rewards.components.obstacle_intrusion import ObstacleIntrusion
        from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction

        comp = ObstacleIntrusion()
        # A triangle around (39.5, 116.5)
        comp.set_obstacles([[(39.4, 116.4), (39.6, 116.4), (39.5, 116.6)]])
        state = AircraftState(id="AC000", lat=39.5, lon=116.5, alt=35000, hdg=90, tas=150, vs=0)
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)
        reward = comp.compute("AC000", state, action, state, {"AC000": state})
        assert reward == -5.0

    def test_intrusion_accumulates(self) -> None:
        """If inside multiple polygons, penalty accumulates."""
        from bluesky_pettingzoo.rewards.components.obstacle_intrusion import ObstacleIntrusion
        from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction

        comp = ObstacleIntrusion()
        # Two overlapping triangles around (39.5, 116.5)
        poly = [(39.4, 116.4), (39.6, 116.4), (39.5, 116.6)]
        comp.set_obstacles([poly, poly])
        state = AircraftState(id="AC000", lat=39.5, lon=116.5, alt=35000, hdg=90, tas=150, vs=0)
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)
        reward = comp.compute("AC000", state, action, state, {"AC000": state})
        assert reward == -10.0

    def test_is_intruded_true(self) -> None:
        from bluesky_pettingzoo.rewards.components.obstacle_intrusion import ObstacleIntrusion
        from bluesky_pettingzoo.utils.types import AircraftState

        comp = ObstacleIntrusion()
        comp.set_obstacles([[(39.4, 116.4), (39.6, 116.4), (39.5, 116.6)]])
        state = AircraftState(id="AC000", lat=39.5, lon=116.5, alt=35000, hdg=90, tas=150, vs=0)
        assert comp.is_intruded(state) is True

    def test_is_intruded_false(self) -> None:
        from bluesky_pettingzoo.rewards.components.obstacle_intrusion import ObstacleIntrusion
        from bluesky_pettingzoo.utils.types import AircraftState

        comp = ObstacleIntrusion()
        comp.set_obstacles([[(40.0, 117.0), (40.1, 117.0), (40.05, 117.1)]])
        state = AircraftState(id="AC000", lat=39.0, lon=116.0, alt=35000, hdg=90, tas=150, vs=0)
        assert comp.is_intruded(state) is False

    def test_reset_clears_obstacles(self) -> None:
        from bluesky_pettingzoo.rewards.components.obstacle_intrusion import ObstacleIntrusion

        comp = ObstacleIntrusion()
        comp.set_obstacles([[(39.4, 116.4), (39.6, 116.4), (39.5, 116.6)]])
        comp.reset()
        assert comp._obstacles == []

    def test_custom_penalty(self) -> None:
        from bluesky_pettingzoo.rewards.components.obstacle_intrusion import ObstacleIntrusion
        from bluesky_pettingzoo.utils.types import AircraftState, DiscreteAction

        comp = ObstacleIntrusion(penalty=-10.0)
        comp.set_obstacles([[(39.4, 116.4), (39.6, 116.4), (39.5, 116.6)]])
        state = AircraftState(id="AC000", lat=39.5, lon=116.5, alt=35000, hdg=90, tas=150, vs=0)
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)
        reward = comp.compute("AC000", state, action, state, {"AC000": state})
        assert reward == -10.0
