"""Tests for renderer decoupling via RendererDataSource Protocol (arch-003)."""

from __future__ import annotations

from unittest.mock import MagicMock

from bluesky_pettingzoo.rendering.base_renderer import BaseRenderer
from bluesky_pettingzoo.utils.protocols import RendererDataSource


class TestRendererDataSourceProtocol:
    """Test RendererDataSource Protocol definition."""

    def test_protocol_exists(self) -> None:
        """RendererDataSource Protocol should be defined."""
        assert hasattr(RendererDataSource, "get_aircraft_states")

    def test_protocol_methods(self) -> None:
        """Protocol should define required methods."""
        assert callable(getattr(RendererDataSource, "get_aircraft_states", None))
        assert callable(getattr(RendererDataSource, "get_waypoints", None))
        assert callable(getattr(RendererDataSource, "get_step_count", None))
        assert callable(getattr(RendererDataSource, "get_active_agents", None))


class TestRendererDoesNotAccessEnv:
    """Verify renderers don't directly access env internals."""

    def _check_renderer_no_env_access(self, renderer_cls: type) -> None:
        """Check that renderer class doesn't have env attribute."""
        import inspect

        source = inspect.getsource(renderer_cls)
        # Check for direct env access patterns
        assert "self.env" not in source, f"{renderer_cls.__name__} accesses self.env"
        assert "self.pz_env" not in source, f"{renderer_cls.__name__} accesses self.pz_env"

    def test_base_renderer_no_env(self) -> None:
        """BaseRenderer should not access env."""
        self._check_renderer_no_env_access(BaseRenderer)

    def test_horizontal_cr_renderer_no_env(self) -> None:
        """HorizontalCRRenderer should not access env."""
        from bluesky_pettingzoo.rendering.horizontal_cr_renderer import HorizontalCRRenderer

        self._check_renderer_no_env_access(HorizontalCRRenderer)

    def test_vertical_cr_renderer_no_env(self) -> None:
        """VerticalCRRenderer should not access env."""
        from bluesky_pettingzoo.rendering.vertical_cr_renderer import VerticalCRRenderer

        self._check_renderer_no_env_access(VerticalCRRenderer)

    def test_descent_renderer_no_env(self) -> None:
        """DescentRenderer should not access env."""
        from bluesky_pettingzoo.rendering.descent_renderer import DescentRenderer

        self._check_renderer_no_env_access(DescentRenderer)

    def test_merge_renderer_no_env(self) -> None:
        """MergeRenderer should not access env."""
        from bluesky_pettingzoo.rendering.merge_renderer import MergeRenderer

        self._check_renderer_no_env_access(MergeRenderer)

    def test_sector_cr_renderer_no_env(self) -> None:
        """SectorCRRenderer should not access env."""
        from bluesky_pettingzoo.rendering.sector_cr_renderer import SectorCRRenderer

        self._check_renderer_no_env_access(SectorCRRenderer)

    def test_star_approach_renderer_no_env(self) -> None:
        """StarApproachRenderer should not access env."""
        from bluesky_pettingzoo.rendering.star_approach_renderer import StarApproachRenderer

        self._check_renderer_no_env_access(StarApproachRenderer)

    def test_waypoint_nav_renderer_no_env(self) -> None:
        """WaypointNavRenderer should not access env."""
        from bluesky_pettingzoo.rendering.waypoint_nav_renderer import WaypointNavRenderer

        self._check_renderer_no_env_access(WaypointNavRenderer)

    def test_plan_waypoint_renderer_no_env(self) -> None:
        """PlanWaypointRenderer should not access env."""
        from bluesky_pettingzoo.rendering.plan_waypoint_renderer import PlanWaypointRenderer

        self._check_renderer_no_env_access(PlanWaypointRenderer)

    def test_static_obstacle_renderer_no_env(self) -> None:
        """StaticObstacleRenderer should not access env."""
        from bluesky_pettingzoo.rendering.static_obstacle_renderer import StaticObstacleRenderer

        self._check_renderer_no_env_access(StaticObstacleRenderer)

    def test_sector_capacity_renderer_no_env(self) -> None:
        """SectorCapacityRenderer should not access env."""
        from bluesky_pettingzoo.rendering.sector_capacity_renderer import SectorCapacityRenderer

        self._check_renderer_no_env_access(SectorCapacityRenderer)

    def test_route_nav_renderer_no_env(self) -> None:
        """RouteNavRenderer should not access env."""
        from bluesky_pettingzoo.rendering.route_nav_renderer import RouteNavRenderer

        self._check_renderer_no_env_access(RouteNavRenderer)


class TestEnvRendererAdapter:
    """Test EnvRendererAdapter implements RendererDataSource."""

    def test_adapter_implements_protocol(self) -> None:
        """EnvRendererAdapter should satisfy RendererDataSource Protocol."""
        from bluesky_pettingzoo.envs.parallel_env import EnvRendererAdapter

        # Create mock dependencies
        obs_builder = MagicMock()
        obs_builder.get_waypoints_for_render.return_value = {}
        step_count = 0
        agents = ["AC000", "AC001"]

        adapter = EnvRendererAdapter(
            obs_builder=obs_builder,
            step_count=step_count,
            agents=agents,
        )

        # Verify it satisfies the protocol
        assert isinstance(adapter, RendererDataSource)

    def test_adapter_methods(self) -> None:
        """EnvRendererAdapter methods should return correct data."""
        from bluesky_pettingzoo.envs.parallel_env import EnvRendererAdapter

        obs_builder = MagicMock()
        obs_builder.get_waypoints_for_render.return_value = {"AC000": {"lat": 40.0, "lon": 117.0}}
        step_count = 42
        agents = ["AC000", "AC001"]

        adapter = EnvRendererAdapter(
            obs_builder=obs_builder,
            step_count=step_count,
            agents=agents,
        )

        assert adapter.get_step_count() == 42
        assert adapter.get_active_agents() == ["AC000", "AC001"]
        assert adapter.get_waypoints() == {"AC000": {"lat": 40.0, "lon": 117.0}}
