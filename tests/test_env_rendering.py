"""Tests for environment rendering integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv


class TestRenderModeParameter:
    """Verify render_mode parameter handling."""

    def test_default_render_mode_none(self):
        env = BlueSkyMARLEnv.__new__(BlueSkyMARLEnv)
        env._render_mode = None
        assert env._render_mode is None

    def test_render_mode_human(self):
        env = BlueSkyMARLEnv.__new__(BlueSkyMARLEnv)
        env._render_mode = "human"
        assert env._render_mode == "human"


class TestRendererInit:
    """Verify renderer is initialized when render_mode='human'."""

    @patch("bluesky_pettingzoo.envs.parallel_env.BlueSkyWrapper")
    @patch("bluesky_pettingzoo.envs.parallel_env.BaseScenario")
    def test_renderer_created_for_horizontal_cr(self, mock_scenario_cls, mock_wrapper_cls):
        """When render_mode='human', env should store a renderer instance."""
        mock_scenario = MagicMock()
        mock_scenario.name = "HorizontalCR"
        mock_scenario.num_aircraft = 5
        mock_scenario.agents = ["AC000", "AC001", "AC002", "AC003", "AC004"]
        mock_scenario.possible_agents = mock_scenario.agents
        mock_scenario_cls.return_value = mock_scenario

        env = BlueSkyMARLEnv.__new__(BlueSkyMARLEnv)
        env._render_mode = "human"
        env._renderer = None
        env._scenario = mock_scenario
        env._airspace = {"lat_min": 39.0, "lat_max": 41.0, "lon_min": 116.0, "lon_max": 118.0}
        env._init_renderer()

        # Should have set a renderer
        assert env._renderer is not None

    @patch("bluesky_pettingzoo.envs.parallel_env.BlueSkyWrapper")
    @patch("bluesky_pettingzoo.envs.parallel_env.BaseScenario")
    def test_no_renderer_when_render_mode_none(self, mock_scenario_cls, mock_wrapper_cls):
        """When render_mode=None, renderer should stay None."""
        env = BlueSkyMARLEnv.__new__(BlueSkyMARLEnv)
        env._render_mode = None
        env._renderer = None
        env._scenario = None
        env._init_renderer()
        assert env._renderer is None
