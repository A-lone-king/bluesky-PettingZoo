"""Tests for LNAV integration (nav-lnav-001).

Verifies that wrapper LNAV methods and scenario configure_npc_navigation
work correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestWrapperLnavMethods:
    """Test BlueSkyWrapper LNAV methods."""

    def test_set_origin(self) -> None:
        """set_origin sends correct BlueSky command."""
        from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper

        with patch("bluesky_pettingzoo.bluesky.wrapper.bs") as mock_bs:
            wrapper = BlueSkyWrapper.__new__(BlueSkyWrapper)
            wrapper.set_origin("AC000", 40.0, 116.0)
            mock_bs.stack.stack.assert_called_once_with("ORIG AC000 40.0 116.0")

    def test_set_destination(self) -> None:
        """set_destination sends correct BlueSky command."""
        from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper

        with patch("bluesky_pettingzoo.bluesky.wrapper.bs") as mock_bs:
            wrapper = BlueSkyWrapper.__new__(BlueSkyWrapper)
            wrapper.set_destination("AC000", 41.0, 117.0)
            mock_bs.stack.stack.assert_called_once_with("DEST AC000 41.0 117.0")

    def test_add_waypoint(self) -> None:
        """add_waypoint sends correct BlueSky command."""
        from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper

        with patch("bluesky_pettingzoo.bluesky.wrapper.bs") as mock_bs:
            wrapper = BlueSkyWrapper.__new__(BlueSkyWrapper)
            wrapper.add_waypoint("AC000", 40.5, 116.5)
            mock_bs.stack.stack.assert_called_once_with("ADDWPT AC000 40.5 116.5")

    def test_enable_lnav(self) -> None:
        """enable_lnav sends correct BlueSky command."""
        from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper

        with patch("bluesky_pettingzoo.bluesky.wrapper.bs") as mock_bs:
            wrapper = BlueSkyWrapper.__new__(BlueSkyWrapper)
            wrapper.enable_lnav("AC000")
            mock_bs.stack.stack.assert_called_once_with("LNAV AC000 ON")

    def test_disable_lnav(self) -> None:
        """disable_lnav sends correct BlueSky command."""
        from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper

        with patch("bluesky_pettingzoo.bluesky.wrapper.bs") as mock_bs:
            wrapper = BlueSkyWrapper.__new__(BlueSkyWrapper)
            wrapper.disable_lnav("AC000")
            mock_bs.stack.stack.assert_called_once_with("LNAV AC000 OFF")


class TestRouteNavConfigureNavigation:
    """Test RouteNavScenario.configure_npc_navigation."""

    def test_sends_lnav_commands(self) -> None:
        """configure_npc_navigation sends LNAV ON for each aircraft."""
        from bluesky_pettingzoo.envs.scenarios.route_nav import RouteNavScenario

        scenario = RouteNavScenario(num_aircraft=3)
        import numpy as np

        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)

        mock_wrapper = MagicMock()
        commands = scenario.configure_npc_navigation(mock_wrapper)

        # Should send reso off + LNAV commands for each aircraft
        assert "reso off" in commands
        lnav_cmds = [c for c in commands if "LNAV" in c]
        assert len(lnav_cmds) == 3
        for cmd in lnav_cmds:
            assert "ON" in cmd


class TestWaypointNavConfigureNavigation:
    """Test WaypointNavScenario.configure_npc_navigation."""

    def test_sends_lnav_commands(self) -> None:
        """configure_npc_navigation sends LNAV ON for each aircraft."""
        from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario

        scenario = WaypointNavScenario(num_aircraft=3)
        import numpy as np

        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        scenario.setup(rng, bounds)

        mock_wrapper = MagicMock()
        commands = scenario.configure_npc_navigation(mock_wrapper)

        # Should send reso off + LNAV commands for each aircraft
        assert "reso off" in commands
        lnav_cmds = [c for c in commands if "LNAV" in c]
        assert len(lnav_cmds) == 3
        for cmd in lnav_cmds:
            assert "ON" in cmd
