"""Tests for scenario-specific renderers — instantiation and basic rendering."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestRendererImports:
    """All renderers can be imported and instantiated."""

    def test_waypoint_nav_renderer_import(self):
        from bluesky_pettingzoo.rendering.waypoint_nav_renderer import WaypointNavRenderer

        r = WaypointNavRenderer()
        assert r is not None

    def test_plan_waypoint_renderer_import(self):
        from bluesky_pettingzoo.rendering.plan_waypoint_renderer import PlanWaypointRenderer

        r = PlanWaypointRenderer()
        assert r is not None

    def test_merge_renderer_import(self):
        from bluesky_pettingzoo.rendering.merge_renderer import MergeRenderer

        r = MergeRenderer()
        assert r is not None

    def test_descent_renderer_import(self):
        from bluesky_pettingzoo.rendering.descent_renderer import DescentRenderer

        r = DescentRenderer()
        assert r is not None

    def test_static_obstacle_renderer_import(self):
        from bluesky_pettingzoo.rendering.static_obstacle_renderer import StaticObstacleRenderer

        r = StaticObstacleRenderer()
        assert r is not None

    def test_sector_capacity_renderer_import(self):
        from bluesky_pettingzoo.rendering.sector_capacity_renderer import SectorCapacityRenderer

        r = SectorCapacityRenderer()
        assert r is not None

    def test_route_nav_renderer_import(self):
        from bluesky_pettingzoo.rendering.route_nav_renderer import RouteNavRenderer

        r = RouteNavRenderer()
        assert r is not None


class TestRendererBounds:
    """All renderers support set_bounds."""

    def test_waypoint_nav_set_bounds(self):
        from bluesky_pettingzoo.rendering.waypoint_nav_renderer import WaypointNavRenderer

        r = WaypointNavRenderer()
        r.set_bounds({"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0})
        assert r._bounds["lat_min"] == 39.0

    def test_plan_waypoint_set_bounds(self):
        from bluesky_pettingzoo.rendering.plan_waypoint_renderer import PlanWaypointRenderer

        r = PlanWaypointRenderer()
        r.set_bounds({"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0})
        assert r._bounds["lat_max"] == 41.0

    def test_merge_set_bounds(self):
        from bluesky_pettingzoo.rendering.merge_renderer import MergeRenderer

        r = MergeRenderer()
        r.set_bounds({"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0})
        assert r._bounds["lon_min"] == 116.0

    def test_descent_set_bounds(self):
        from bluesky_pettingzoo.rendering.descent_renderer import DescentRenderer

        r = DescentRenderer()
        r.set_bounds({"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0})
        assert r._bounds["lon_max"] == 118.0

    def test_static_obstacle_set_bounds(self):
        from bluesky_pettingzoo.rendering.static_obstacle_renderer import StaticObstacleRenderer

        r = StaticObstacleRenderer()
        r.set_bounds({"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0})
        assert r._bounds["lat_min"] == 39.0

    def test_sector_capacity_set_bounds(self):
        from bluesky_pettingzoo.rendering.sector_capacity_renderer import SectorCapacityRenderer

        r = SectorCapacityRenderer()
        r.set_bounds({"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0})
        assert r._bounds["lat_max"] == 41.0

    def test_route_nav_set_bounds(self):
        from bluesky_pettingzoo.rendering.route_nav_renderer import RouteNavRenderer

        r = RouteNavRenderer()
        r.set_bounds({"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0})
        assert r._bounds["lon_min"] == 116.0


class TestRendererRenderFrame:
    """Test render_frame with mocked drawing functions."""

    def _make_state(self, lat=40.0, lon=117.0, alt=35000.0, hdg=90.0, tas=450.0, vs=0.0):
        state = MagicMock()
        state.lat = lat
        state.lon = lon
        state.alt = alt
        state.hdg = hdg
        state.tas = tas
        state.vs = vs
        return state

    @patch("bluesky_pettingzoo.rendering.waypoint_nav_renderer.draw_aircraft")
    @patch("bluesky_pettingzoo.rendering.waypoint_nav_renderer.draw_nmac_circle")
    @patch("bluesky_pettingzoo.rendering.waypoint_nav_renderer.draw_waypoint")
    def test_waypoint_nav_renders_without_error(self, mock_wp, mock_nmac, mock_ac):
        from bluesky_pettingzoo.rendering.waypoint_nav_renderer import WaypointNavRenderer

        r = WaypointNavRenderer()
        r._initialized = True
        r._screen = MagicMock()
        r._font = MagicMock()
        r.flip = MagicMock()
        states = {"AC000": self._make_state()}
        waypoints = {"AC000": {"lat": 40.1, "lon": 117.1}}
        r.render_frame(states, waypoints=waypoints, step=1)
        mock_ac.assert_called_once()
        mock_nmac.assert_called_once()
        mock_wp.assert_called_once()

    @patch("bluesky_pettingzoo.rendering.plan_waypoint_renderer.pygame")
    @patch("bluesky_pettingzoo.rendering.plan_waypoint_renderer.draw_aircraft")
    @patch("bluesky_pettingzoo.rendering.plan_waypoint_renderer.draw_nmac_circle")
    @patch("bluesky_pettingzoo.rendering.plan_waypoint_renderer.draw_waypoint")
    def test_plan_waypoint_renders_without_error(self, mock_wp, mock_nmac, mock_ac, mock_pygame):
        from bluesky_pettingzoo.rendering.plan_waypoint_renderer import PlanWaypointRenderer

        r = PlanWaypointRenderer()
        r._initialized = True
        r._screen = MagicMock()
        r._font = MagicMock()
        r.flip = MagicMock()
        states = {"AC000": self._make_state()}
        waypoints = [
            {"lat": 40.0, "lon": 117.0, "reached": True},
            {"lat": 40.1, "lon": 117.1, "reached": False},
        ]
        r.render_frame(states, waypoints=waypoints, step=1)
        assert mock_wp.call_count == 2
        mock_pygame.draw.lines.assert_called_once()

    @patch("bluesky_pettingzoo.rendering.merge_renderer.draw_aircraft")
    @patch("bluesky_pettingzoo.rendering.merge_renderer.draw_nmac_circle")
    @patch("bluesky_pettingzoo.rendering.merge_renderer.draw_waypoint")
    def test_merge_renders_without_error(self, mock_wp, mock_nmac, mock_ac):
        from bluesky_pettingzoo.rendering.merge_renderer import MergeRenderer

        r = MergeRenderer()
        r._initialized = True
        r._screen = MagicMock()
        r._font = MagicMock()
        r.flip = MagicMock()
        states = {
            "AC000": self._make_state(lat=40.0, lon=117.0),
            "AC001": self._make_state(lat=40.1, lon=117.1),
        }
        waypoints = {
            "AC000": {"lat": 40.05, "lon": 117.05},
            "AC001": {"lat": 40.05, "lon": 117.05},
        }
        r.render_frame(
            states,
            waypoints=waypoints,
            step=1,
            info={"controllable": ["AC000"], "background": ["AC001"]},
        )
        assert mock_ac.call_count == 2

    @patch("bluesky_pettingzoo.rendering.descent_renderer.pygame")
    @patch("bluesky_pettingzoo.rendering.descent_renderer.draw_aircraft")
    @patch("bluesky_pettingzoo.rendering.descent_renderer.draw_nmac_circle")
    @patch("bluesky_pettingzoo.rendering.descent_renderer.draw_waypoint")
    def test_descent_renders_without_error(self, mock_wp, mock_nmac, mock_ac, mock_pygame):
        from bluesky_pettingzoo.rendering.descent_renderer import DescentRenderer

        r = DescentRenderer()
        r._initialized = True
        r._screen = MagicMock()
        r._font = MagicMock()
        r.flip = MagicMock()
        states = {"AC000": self._make_state(alt=30000.0)}
        waypoints = {"AC000": {"lat": 40.0, "lon": 117.0, "alt": 3000.0}}
        r.render_frame(states, waypoints=waypoints, step=1)
        mock_ac.assert_called_once()
        mock_wp.assert_called_once()
        mock_pygame.draw.line.assert_called()

    @patch("bluesky_pettingzoo.rendering.static_obstacle_renderer.draw_aircraft")
    @patch("bluesky_pettingzoo.rendering.static_obstacle_renderer.draw_nmac_circle")
    @patch("bluesky_pettingzoo.rendering.static_obstacle_renderer.draw_sector_polygon")
    @patch("bluesky_pettingzoo.rendering.static_obstacle_renderer.draw_waypoint")
    def test_static_obstacle_renders_without_error(self, mock_wp, mock_poly, mock_nmac, mock_ac):
        from bluesky_pettingzoo.rendering.static_obstacle_renderer import StaticObstacleRenderer

        r = StaticObstacleRenderer()
        r._initialized = True
        r._screen = MagicMock()
        r._font = MagicMock()
        r.flip = MagicMock()
        states = {"AC000": self._make_state()}
        obstacles = [[(39.5, 116.5), (39.5, 117.5), (40.5, 117.5), (40.5, 116.5)]]
        r.render_frame(states, step=1, info={"obstacles": obstacles})
        mock_poly.assert_called_once()
        mock_ac.assert_called_once()

    @patch("bluesky_pettingzoo.rendering.sector_capacity_renderer.draw_aircraft")
    @patch("bluesky_pettingzoo.rendering.sector_capacity_renderer.draw_nmac_circle")
    @patch("bluesky_pettingzoo.rendering.sector_capacity_renderer.draw_sector_polygon")
    @patch("bluesky_pettingzoo.rendering.sector_capacity_renderer.draw_waypoint")
    def test_sector_capacity_renders_without_error(self, mock_wp, mock_poly, mock_nmac, mock_ac):
        from bluesky_pettingzoo.rendering.sector_capacity_renderer import SectorCapacityRenderer

        r = SectorCapacityRenderer()
        r._initialized = True
        r._screen = MagicMock()
        r._font = MagicMock()
        r.flip = MagicMock()
        states = {"AC000": self._make_state()}
        sectors = [
            {"id": "sector_0", "bounds": [[39.0, 116.0], [41.0, 117.0]], "capacity": 5},
            {"id": "sector_1", "bounds": [[39.0, 117.0], [41.0, 118.0]], "capacity": 3},
        ]
        r.render_frame(states, step=1, info={"sectors": sectors})
        assert mock_poly.call_count == 2

    @patch("bluesky_pettingzoo.rendering.route_nav_renderer.pygame")
    @patch("bluesky_pettingzoo.rendering.route_nav_renderer.draw_aircraft")
    @patch("bluesky_pettingzoo.rendering.route_nav_renderer.draw_nmac_circle")
    @patch("bluesky_pettingzoo.rendering.route_nav_renderer.draw_waypoint")
    def test_route_nav_renders_without_error(self, mock_wp, mock_nmac, mock_ac, mock_pygame):
        from bluesky_pettingzoo.rendering.route_nav_renderer import RouteNavRenderer

        r = RouteNavRenderer()
        r._initialized = True
        r._screen = MagicMock()
        r._font = MagicMock()
        r.flip = MagicMock()
        states = {"AC000": self._make_state()}
        routes = {
            "AC000": {
                "waypoints": [
                    {"lat": 40.0, "lon": 117.0},
                    {"lat": 40.1, "lon": 117.1},
                ]
            }
        }
        r.render_frame(states, step=1, info={"routes": routes})
        mock_ac.assert_called_once()
        mock_pygame.draw.lines.assert_called_once()


class TestRendererInheritance:
    """All renderers inherit from BaseRenderer."""

    def test_all_renderers_are_base_subclasses(self):
        from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer
        from bluesky_pettingzoo.rendering.descent_renderer import DescentRenderer
        from bluesky_pettingzoo.rendering.merge_renderer import MergeRenderer
        from bluesky_pettingzoo.rendering.plan_waypoint_renderer import PlanWaypointRenderer
        from bluesky_pettingzoo.rendering.route_nav_renderer import RouteNavRenderer
        from bluesky_pettingzoo.rendering.sector_capacity_renderer import SectorCapacityRenderer
        from bluesky_pettingzoo.rendering.static_obstacle_renderer import StaticObstacleRenderer
        from bluesky_pettingzoo.rendering.waypoint_nav_renderer import WaypointNavRenderer

        for cls in [
            WaypointNavRenderer,
            PlanWaypointRenderer,
            MergeRenderer,
            DescentRenderer,
            StaticObstacleRenderer,
            SectorCapacityRenderer,
            RouteNavRenderer,
        ]:
            assert issubclass(cls, BaseRenderer), f"{cls.__name__} must inherit BaseRenderer"


class TestParallelEnvRendererMapping:
    """parallel_env._init_renderer maps all scenarios to renderers."""

    def test_renderer_map_has_all_scenarios(self):
        import pathlib

        env_file = (
            pathlib.Path(__file__).resolve().parent.parent
            / "src"
            / "bluesky_pettingzoo"
            / "envs"
            / "parallel_env.py"
        )
        source = env_file.read_text(encoding="utf-8")

        expected_names = [
            "HorizontalCR",
            "VerticalCR",
            "SectorCR",
            "WaypointNav",
            "PlanWaypoint",
            "Merge",
            "Descent",
            "StaticObstacle",
            "SectorCapacity",
            "RouteNav",
        ]
        for name in expected_names:
            assert f'"{name}"' in source, f"Missing renderer mapping for {name}"
