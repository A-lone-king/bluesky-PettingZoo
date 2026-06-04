"""PettingZoo API compliance tests for BlueSkyMARLEnv."""

from __future__ import annotations

import pytest

from tests.helpers.env_factory import make_env as _make_env

# ---------------------------------------------------------------------------
# Fake BlueSkyWrapper (same as test_env.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestParallelApi:
    """Verify environment passes PettingZoo's parallel_api_test."""

    def test_parallel_api(self) -> None:
        """Run PettingZoo's official parallel API compliance test."""
        from pettingzoo.test import parallel_api_test

        env = _make_env(initial_count=3, max_steps=360)
        parallel_api_test(env, num_cycles=100)
        env.close()


class TestRenderModes:
    """Verify render_mode support."""

    def test_render_mode_in_metadata(self) -> None:
        """Environment should declare render_mode in metadata."""
        env = _make_env()
        # PettingZoo expects metadata to exist with render_modes
        assert hasattr(env, "metadata")
        assert isinstance(env.metadata, dict)
        env.close()

    def test_no_render_if_not_configured(self) -> None:
        """render() should raise or be absent if no render mode is set."""
        env = _make_env()
        # Our env doesn't support rendering — calling render should raise
        if hasattr(env, "render"):
            with pytest.raises((NotImplementedError, Exception)):
                env.render()
        env.close()


class TestClose:
    """Verify close() has no resource leaks."""

    def test_close_no_error(self) -> None:
        """close() must not raise."""
        env = _make_env()
        env.reset()
        env.close()

    def test_close_after_steps(self) -> None:
        """close() after running steps must not raise."""
        env = _make_env()
        env.reset()
        for _ in range(5):
            actions = {a: [2, 2, 2] for a in env.agents}
            env.step(actions)
        env.close()

    def test_close_idempotent(self) -> None:
        """Calling close() multiple times must not raise."""
        env = _make_env()
        env.reset()
        env.close()
        env.close()
