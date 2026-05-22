"""Tests for NoisyObservationWrapper."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from gymnasium import spaces

from bluesky_pettingzoo.wrappers.noisy_observation import NoisyObservationWrapper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_env(
    agents: list[str] | None = None,
    obs_shape: tuple[int, ...] = (6,),
) -> MagicMock:
    """Create a mock ParallelEnv with Dict observation space."""
    if agents is None:
        agents = ["AC001", "AC002", "AC003"]

    env = MagicMock()
    env.agents = list(agents)
    env.possible_agents = list(agents)
    env.observation_space = MagicMock(return_value=spaces.Dict({
        "self_state": spaces.Box(-1.0, 1.0, shape=obs_shape, dtype=np.float32),
        "other_aircraft": spaces.Box(-1.0, 1.0, shape=(5, 7), dtype=np.float32),
    }))
    env.action_space = MagicMock(return_value=spaces.MultiDiscrete([5, 5, 5]))

    # Default reset returns
    obs = {
        aid: {
            "self_state": np.zeros(obs_shape, dtype=np.float32),
            "other_aircraft": np.zeros((5, 7), dtype=np.float32),
        }
        for aid in agents
    }
    infos = {aid: {} for aid in agents}
    env.reset = MagicMock(return_value=(obs, infos))

    # Default step returns
    rewards = {aid: 0.0 for aid in agents}
    terminations = {aid: False for aid in agents}
    truncations = {aid: False for aid in agents}
    env.step = MagicMock(return_value=(obs, rewards, terminations, truncations, infos))

    return env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoiseChangesObservation:
    def test_noise_changes_observation(self) -> None:
        """Noise should alter observation values."""
        env = _make_mock_env()
        wrapper = NoisyObservationWrapper(env, noise_level=0.5, seed=42)

        obs, _ = wrapper.reset()
        for agent_id in env.agents:
            assert not np.allclose(obs[agent_id]["self_state"], 0.0), (
                f"self_state for {agent_id} should be non-zero after noise"
            )


class TestNoiseMagnitudeProportional:
    def test_noise_magnitude_proportional_to_level(self) -> None:
        """Larger noise_level should produce larger deviations."""
        env1 = _make_mock_env()
        wrapper1 = NoisyObservationWrapper(env1, noise_level=0.01, seed=42)
        obs1, _ = wrapper1.reset()

        env2 = _make_mock_env()
        wrapper2 = NoisyObservationWrapper(env2, noise_level=1.0, seed=42)
        obs2, _ = wrapper2.reset()

        dev_small = np.abs(obs1["AC001"]["self_state"]).mean()
        dev_large = np.abs(obs2["AC001"]["self_state"]).mean()
        assert dev_large > dev_small, "Larger noise_level should produce larger deviations"


class TestSameSeedSameNoise:
    def test_same_seed_same_noise(self) -> None:
        """Same seed should produce identical noise."""
        env1 = _make_mock_env()
        wrapper1 = NoisyObservationWrapper(env1, noise_level=0.3, seed=123)
        obs1, _ = wrapper1.reset()

        env2 = _make_mock_env()
        wrapper2 = NoisyObservationWrapper(env2, noise_level=0.3, seed=123)
        obs2, _ = wrapper2.reset()

        for agent_id in env1.agents:
            np.testing.assert_array_equal(
                obs1[agent_id]["self_state"],
                obs2[agent_id]["self_state"],
                err_msg=f"Same seed should produce identical noise for {agent_id}",
            )


class TestAgentsDelegated:
    def test_agents_delegated(self) -> None:
        """agents and possible_agents should be delegated to wrapped env."""
        env = _make_mock_env(agents=["A", "B"])
        wrapper = NoisyObservationWrapper(env, noise_level=0.1)

        assert wrapper.agents == ["A", "B"]
        assert wrapper.possible_agents == ["A", "B"]


class TestObservationSpaceDelegated:
    def test_observation_space_delegated(self) -> None:
        """observation_space should be delegated to wrapped env."""
        env = _make_mock_env()
        wrapper = NoisyObservationWrapper(env, noise_level=0.1)

        space = wrapper.observation_space("AC001")
        assert isinstance(space, spaces.Dict)
        assert "self_state" in space.spaces


class TestActionSpaceDelegated:
    def test_action_space_delegated(self) -> None:
        """action_space should be delegated to wrapped env."""
        env = _make_mock_env()
        wrapper = NoisyObservationWrapper(env, noise_level=0.1)

        space = wrapper.action_space("AC001")
        assert isinstance(space, spaces.MultiDiscrete)


class TestTerminationsUnchanged:
    def test_terminations_unchanged(self) -> None:
        """terminations and truncations should pass through unmodified."""
        env = _make_mock_env()
        wrapper = NoisyObservationWrapper(env, noise_level=0.1, seed=42)
        wrapper.reset()

        env.step.return_value = (
            env.step.return_value[0],  # obs
            {"AC001": -1.0, "AC002": 0.0, "AC003": 0.5},  # rewards
            {"AC001": True, "AC002": False, "AC003": False},  # terminations
            {"AC001": False, "AC002": True, "AC003": False},  # truncations
            {"AC001": {}, "AC002": {}, "AC003": {}},  # infos
        )

        _, rewards, terms, truncs, _ = wrapper.step({"AC001": [2, 2, 2]})

        assert terms == {"AC001": True, "AC002": False, "AC003": False}
        assert truncs == {"AC001": False, "AC002": True, "AC003": False}
        assert rewards == {"AC001": -1.0, "AC002": 0.0, "AC003": 0.5}


class TestNoNoiseLevelZero:
    def test_no_noise_level_zero(self) -> None:
        """noise_level=0 should not change observations."""
        env = _make_mock_env()
        wrapper = NoisyObservationWrapper(env, noise_level=0.0, seed=42)

        obs, _ = wrapper.reset()
        for agent_id in env.agents:
            np.testing.assert_array_equal(
                obs[agent_id]["self_state"],
                np.zeros((6,), dtype=np.float32),
                err_msg="noise_level=0 should not change observations",
            )


class TestDictObservationNoise:
    def test_dict_observation_noise(self) -> None:
        """All Box values in Dict observation should get noise."""
        env = _make_mock_env()
        wrapper = NoisyObservationWrapper(env, noise_level=0.5, seed=42)

        obs, _ = wrapper.reset()
        for agent_id in env.agents:
            assert not np.allclose(obs[agent_id]["self_state"], 0.0)
            assert not np.allclose(obs[agent_id]["other_aircraft"], 0.0)


class TestNonNdarrayValuesSkipped:
    def test_non_ndarray_values_skipped(self) -> None:
        """Non-ndarray values in observation dict should be passed through unchanged."""
        env = _make_mock_env()
        # Override observation to include a non-ndarray value
        obs_with_text = {
            aid: {
                "self_state": np.zeros((6,), dtype=np.float32),
                "text_status": "safe",
            }
            for aid in env.agents
        }
        env.reset = MagicMock(return_value=(obs_with_text, {aid: {} for aid in env.agents}))

        wrapper = NoisyObservationWrapper(env, noise_level=0.5, seed=42)
        obs, _ = wrapper.reset()

        for agent_id in env.agents:
            assert obs[agent_id]["text_status"] == "safe", (
                "Non-ndarray values should be passed through unchanged"
            )


class TestResetAddsNoise:
    def test_reset_adds_noise(self) -> None:
        """reset() should add noise to observations."""
        env = _make_mock_env()
        wrapper = NoisyObservationWrapper(env, noise_level=0.3, seed=42)

        obs, _ = wrapper.reset()
        # At least one agent's self_state should differ from zero
        any_nonzero = any(
            not np.allclose(obs[aid]["self_state"], 0.0) for aid in env.agents
        )
        assert any_nonzero, "reset() should add noise"


class TestStepAddsNoise:
    def test_step_adds_noise(self) -> None:
        """step() should add noise to observations."""
        env = _make_mock_env()
        wrapper = NoisyObservationWrapper(env, noise_level=0.3, seed=42)
        wrapper.reset()

        obs, _, _, _, _ = wrapper.step({aid: [2, 2, 2] for aid in env.agents})
        any_nonzero = any(
            not np.allclose(obs[aid]["self_state"], 0.0) for aid in env.agents
        )
        assert any_nonzero, "step() should add noise"


class TestCloseDelegated:
    def test_close_delegated(self) -> None:
        """close() should be delegated to wrapped env."""
        env = _make_mock_env()
        wrapper = NoisyObservationWrapper(env, noise_level=0.1)
        wrapper.close()
        env.close.assert_called_once()
