"""Tests for WindFieldWrapper."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from gymnasium import spaces

from bluesky_pettingzoo.wrappers.wind_field import WindFieldWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_env(
    agents: list[str] | None = None,
    headings: dict[str, float] | None = None,
) -> MagicMock:
    """Create a mock ParallelEnv with Dict observation space."""
    if agents is None:
        agents = ["AC001", "AC002"]
    if headings is None:
        headings = {"AC001": 90.0, "AC002": 0.0}

    env = MagicMock()
    env.agents = list(agents)
    env.possible_agents = list(agents)
    env.observation_space = MagicMock(return_value=spaces.Dict({
        "self_state": spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32),
    }))
    env.action_space = MagicMock(return_value=spaces.MultiDiscrete([5, 5, 5]))

    # Build observations
    obs = {
        aid: {"self_state": np.zeros(6, dtype=np.float32)}
        for aid in agents
    }
    infos = {aid: {} for aid in agents}
    env.reset = MagicMock(return_value=(obs, infos))

    rewards = {aid: 0.0 for aid in agents}
    terminations = {aid: False for aid in agents}
    truncations = {aid: False for aid in agents}
    env.step = MagicMock(return_value=(obs, rewards, terminations, truncations, infos))

    # Mock aircraft_states property on unwrapped env (public API)
    from bluesky_pettingzoo.utils.types import AircraftState

    env.unwrapped.aircraft_states = {
        acid: AircraftState(
            id=acid, lat=39.0, lon=116.0, alt=35000.0,
            hdg=headings.get(acid, 0.0), tas=450.0, vs=0.0,
        )
        for acid in agents
    }

    return env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAugmentObsAddsWind:
    @patch("bluesky_pettingzoo.wrappers.wind_field.bs")
    def test_augment_obs_adds_wind_keys(self, mock_bs: MagicMock) -> None:
        """augment_obs=True should add wind_u and wind_v to observations."""
        env = _make_mock_env()
        wrapper = WindFieldWrapper(
            env, lat=39.0, lon=116.0, vnorth=10.0, veast=0.0,
            augment_obs=True, seed=42,
        )

        # Mock wind data: pure north wind (10 kt from north)
        mock_bs.traf.wind.getdata = MagicMock(return_value=(10.0, 0.0))

        obs, _ = wrapper.reset()
        for agent_id in env.agents:
            assert "wind_u" in obs[agent_id], f"wind_u missing for {agent_id}"
            assert "wind_v" in obs[agent_id], f"wind_v missing for {agent_id}"


class TestWindBodyFrameConversion:
    @patch("bluesky_pettingzoo.wrappers.wind_field.bs")
    def test_wind_body_frame_heading_90(self, mock_bs: MagicMock) -> None:
        """Heading 90°: north wind becomes crosswind (wind_v), no headwind (wind_u)."""
        env = _make_mock_env(headings={"AC001": 90.0})
        wrapper = WindFieldWrapper(
            env, lat=39.0, lon=116.0, vnorth=10.0, veast=0.0,
            augment_obs=True, seed=42,
        )

        # Pure north wind: wind_n=10, wind_e=0
        mock_bs.traf.wind.getdata = MagicMock(return_value=(10.0, 0.0))

        obs, _ = wrapper.reset()

        # Heading 90° (east): cos(90°)=0, sin(90°)=1
        # wind_u = wind_n * cos(hdg) + wind_e * sin(hdg) = 10*0 + 0*1 = 0
        # wind_v = -wind_n * sin(hdg) + wind_e * cos(hdg) = -10*1 + 0*0 = -10
        # Normalized by MAX_WIND=50
        np.testing.assert_allclose(
            obs["AC001"]["wind_u"], 0.0, atol=1e-6,
            err_msg="wind_u should be ~0 for heading 90 with north wind",
        )
        np.testing.assert_allclose(
            obs["AC001"]["wind_v"], -10.0 / 50.0, atol=1e-6,
            err_msg="wind_v should be -0.2 for heading 90 with north wind",
        )

    @patch("bluesky_pettingzoo.wrappers.wind_field.bs")
    def test_wind_body_frame_heading_0(self, mock_bs: MagicMock) -> None:
        """Heading 0°: north wind becomes headwind (wind_u)."""
        env = _make_mock_env(headings={"AC001": 0.0})
        wrapper = WindFieldWrapper(
            env, lat=39.0, lon=116.0, vnorth=10.0, veast=0.0,
            augment_obs=True, seed=42,
        )

        mock_bs.traf.wind.getdata = MagicMock(return_value=(10.0, 0.0))

        obs, _ = wrapper.reset()

        # Heading 0° (north): cos(0°)=1, sin(0°)=0
        # wind_u = 10*1 + 0*0 = 10
        # wind_v = -10*0 + 0*1 = 0
        np.testing.assert_allclose(
            obs["AC001"]["wind_u"], 10.0 / 50.0, atol=1e-6,
            err_msg="wind_u should be 0.2 for heading 0 with north wind",
        )
        np.testing.assert_allclose(
            obs["AC001"]["wind_v"], 0.0, atol=1e-6,
            err_msg="wind_v should be ~0 for heading 0 with north wind",
        )


class TestNoAugmentObs:
    @patch("bluesky_pettingzoo.wrappers.wind_field.bs")
    def test_no_augment_obs_unchanged(self, mock_bs: MagicMock) -> None:
        """augment_obs=False should not modify observations."""
        env = _make_mock_env()
        wrapper = WindFieldWrapper(
            env, lat=39.0, lon=116.0, vnorth=10.0, veast=0.0,
            augment_obs=False, seed=42,
        )

        obs, _ = wrapper.reset()
        for agent_id in env.agents:
            assert "wind_u" not in obs[agent_id], "wind_u should not be added when augment_obs=False"
            assert "wind_v" not in obs[agent_id], "wind_v should not be added when augment_obs=False"


class TestObservationSpaceExtended:
    @patch("bluesky_pettingzoo.wrappers.wind_field.bs")
    def test_observation_space_includes_wind(self, mock_bs: MagicMock) -> None:
        """observation_space should include wind_u and wind_v when augment_obs=True."""
        env = _make_mock_env()
        wrapper = WindFieldWrapper(
            env, lat=39.0, lon=116.0, vnorth=10.0, veast=0.0,
            augment_obs=True, seed=42,
        )

        space = wrapper.observation_space("AC001")
        assert "wind_u" in space.spaces, "wind_u should be in observation space"
        assert "wind_v" in space.spaces, "wind_v should be in observation space"


class TestStepAddsWind:
    @patch("bluesky_pettingzoo.wrappers.wind_field.bs")
    def test_step_adds_wind(self, mock_bs: MagicMock) -> None:
        """step() should also augment observations with wind."""
        env = _make_mock_env()
        wrapper = WindFieldWrapper(
            env, lat=39.0, lon=116.0, vnorth=10.0, veast=0.0,
            augment_obs=True, seed=42,
        )
        mock_bs.traf.wind.getdata = MagicMock(return_value=(10.0, 0.0))
        wrapper.reset()

        obs, _, _, _, _ = wrapper.step({aid: [2, 2, 2] for aid in env.agents})
        for agent_id in env.agents:
            assert "wind_u" in obs[agent_id], f"wind_u missing in step obs for {agent_id}"
            assert "wind_v" in obs[agent_id], f"wind_v missing in step obs for {agent_id}"


class TestDelegation:
    @patch("bluesky_pettingzoo.wrappers.wind_field.bs")
    def test_agents_delegated(self, mock_bs: MagicMock) -> None:
        """agents should be delegated to wrapped env."""
        env = _make_mock_env(agents=["A", "B"])
        wrapper = WindFieldWrapper(
            env, lat=39.0, lon=116.0, vnorth=10.0, veast=0.0,
            augment_obs=False, seed=42,
        )
        assert wrapper.agents == ["A", "B"]

    @patch("bluesky_pettingzoo.wrappers.wind_field.bs")
    def test_close_delegated(self, mock_bs: MagicMock) -> None:
        """close() should be delegated."""
        env = _make_mock_env()
        wrapper = WindFieldWrapper(
            env, lat=39.0, lon=116.0, vnorth=10.0, veast=0.0,
            augment_obs=False, seed=42,
        )
        wrapper.close()
        env.close.assert_called_once()


class TestRewardsAndTermsUnchanged:
    @patch("bluesky_pettingzoo.wrappers.wind_field.bs")
    def test_rewards_and_terms_unchanged(self, mock_bs: MagicMock) -> None:
        """rewards, terminations, truncations should pass through unmodified."""
        env = _make_mock_env()
        wrapper = WindFieldWrapper(
            env, lat=39.0, lon=116.0, vnorth=10.0, veast=0.0,
            augment_obs=True, seed=42,
        )
        mock_bs.traf.wind.getdata = MagicMock(return_value=(10.0, 0.0))
        wrapper.reset()

        env.step.return_value = (
            env.step.return_value[0],
            {"AC001": -5.0, "AC002": 1.0},
            {"AC001": True, "AC002": False},
            {"AC001": False, "AC002": True},
            {"AC001": {}, "AC002": {}},
        )

        _, rewards, terms, truncs, _ = wrapper.step({"AC001": [2, 2, 2]})
        assert rewards == {"AC001": -5.0, "AC002": 1.0}
        assert terms == {"AC001": True, "AC002": False}
        assert truncs == {"AC001": False, "AC002": True}
