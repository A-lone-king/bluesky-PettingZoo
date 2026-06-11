"""Tests for parallel_env.py exception handling and safe termination fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

from tests.helpers.env_factory import make_env


class TestSafeTerminationFallback:
    """Test _safe_termination_fallback method."""

    def test_fallback_returns_correct_structure(self, tmp_path: Path) -> None:
        """Fallback should return (obs, rewards, terms, truncs, infos) with correct keys."""
        env = make_env(tmp_path, initial_count=3)
        env.reset(seed=42)

        error = RuntimeError("BlueSky engine crashed")
        obs, rewards, terms, truncs, infos = env._safe_termination_fallback(
            actions={"AC000": [2, 2, 2]}, error=error
        )

        # Check structure
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terms, dict)
        assert isinstance(truncs, dict)
        assert isinstance(infos, dict)

        # Check all agents are present
        for agent_id in env.agents:
            assert agent_id in obs
            assert agent_id in rewards
            assert agent_id in terms
            assert agent_id in truncs
            assert agent_id in infos

    def test_fallback_rewards_are_crash_penalty(self, tmp_path: Path) -> None:
        """All agents should receive crash_penalty reward."""
        env = make_env(tmp_path, initial_count=3)
        env.reset(seed=42)

        error = ValueError("Test error")
        _, rewards, _, _, _ = env._safe_termination_fallback(
            actions={}, error=error
        )

        for agent_id in env.agents:
            assert rewards[agent_id] == -100.0  # default crash_penalty

    def test_fallback_rewards_custom_crash_penalty(self, tmp_path: Path) -> None:
        """Should use config crash_penalty if set."""
        config = {"simulation": {"crash_penalty": -200.0}}
        env = make_env(tmp_path, initial_count=2, **config)
        env.reset(seed=42)

        error = RuntimeError("Test")
        _, rewards, _, _, _ = env._safe_termination_fallback(
            actions={}, error=error
        )

        for agent_id in env.agents:
            assert rewards[agent_id] == -200.0

    def test_fallback_terminations_all_true(self, tmp_path: Path) -> None:
        """All agents should be terminated."""
        env = make_env(tmp_path, initial_count=3)
        env.reset(seed=42)

        error = RuntimeError("Test")
        _, _, terms, _, _ = env._safe_termination_fallback(
            actions={}, error=error
        )

        for agent_id in env.agents:
            assert terms[agent_id] is True

    def test_fallback_truncations_all_false(self, tmp_path: Path) -> None:
        """No agents should be truncated (terminated, not truncated)."""
        env = make_env(tmp_path, initial_count=3)
        env.reset(seed=42)

        error = RuntimeError("Test")
        _, _, _, truncs, _ = env._safe_termination_fallback(
            actions={}, error=error
        )

        for agent_id in env.agents:
            assert truncs[agent_id] is False

    def test_fallback_infos_contain_error(self, tmp_path: Path) -> None:
        """Infos should contain error message and type."""
        env = make_env(tmp_path, initial_count=2)
        env.reset(seed=42)

        error = RuntimeError("Engine failure")
        _, _, _, _, infos = env._safe_termination_fallback(
            actions={}, error=error
        )

        for agent_id in env.agents:
            assert "error" in infos[agent_id]
            assert "error_type" in infos[agent_id]
            assert "step" in infos[agent_id]
            assert infos[agent_id]["error"] == "Engine failure"
            assert infos[agent_id]["error_type"] == "RuntimeError"

    def test_fallback_observations_are_default(self, tmp_path: Path) -> None:
        """Observations should be default observations from the space."""
        env = make_env(tmp_path, initial_count=2)
        env.reset(seed=42)

        error = RuntimeError("Test")
        obs, _, _, _, _ = env._safe_termination_fallback(
            actions={}, error=error
        )

        for agent_id in env.agents:
            assert "self_state" in obs[agent_id]
            assert "other_aircraft" in obs[agent_id]
            assert "conflict_state" in obs[agent_id]

    def test_fallback_logs_error(self, tmp_path: Path) -> None:
        """Should log error message."""
        env = make_env(tmp_path, initial_count=2)
        env.reset(seed=42)

        error = RuntimeError("Test error")
        with patch.object(logging, "error") as mock_log:
            env._safe_termination_fallback(actions={}, error=error)
            mock_log.assert_called_once()
            args = mock_log.call_args
            assert "BlueSky engine error" in args[0][0]


class TestStepExceptionHandling:
    """Test step() catches BlueSky exceptions and returns fallback."""

    def test_step_returns_fallback_on_wrapper_error(self, tmp_path: Path) -> None:
        """step() should return safe fallback when wrapper raises exception."""
        env = make_env(tmp_path, initial_count=2)
        env.reset(seed=42)

        actions = {agent_id: [2, 2, 2] for agent_id in env.agents}

        with patch.object(
            env._wrapper, "send_commands_batch", side_effect=RuntimeError("Engine crash")
        ):
            obs, rewards, terms, truncs, infos = env.step(actions)

        # Should return valid structure
        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)

        # All agents should be terminated with crash penalty
        for agent_id in actions:
            if agent_id in terms:
                assert terms[agent_id] is True
                assert rewards[agent_id] == -100.0

    def test_step_returns_fallback_on_step_n_error(self, tmp_path: Path) -> None:
        """step() should return safe fallback when step_n raises exception."""
        env = make_env(tmp_path, initial_count=2)
        env.reset(seed=42)

        actions = {agent_id: [2, 2, 2] for agent_id in env.agents}

        with patch.object(
            env._wrapper, "step_n", side_effect=ValueError("Step failed")
        ):
            obs, rewards, terms, truncs, infos = env.step(actions)

        assert isinstance(obs, dict)
        assert isinstance(rewards, dict)

    def test_step_error_does_not_crash_env(self, tmp_path: Path) -> None:
        """Environment should remain usable after an error in step()."""
        env = make_env(tmp_path, initial_count=2)
        env.reset(seed=42)

        actions = {agent_id: [2, 2, 2] for agent_id in env.agents}

        # First call fails
        with patch.object(
            env._wrapper, "send_commands_batch", side_effect=RuntimeError("Crash")
        ):
            env.step(actions)

        # Second call should work normally (wrapper恢复)
        with patch.object(env._wrapper, "send_commands_batch"):
            with patch.object(env._wrapper, "step_n"):
                with patch.object(
                    env._wrapper, "get_all_aircraft_states", return_value={}
                ):
                    obs, rewards, terms, truncs, infos = env.step(actions)
                    # Should return valid structure without crashing
                    assert isinstance(obs, dict)
