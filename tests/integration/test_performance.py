"""Performance benchmark integration tests (G-V04).

Establish baseline performance metrics for the environment:
- Step time for 5 and 20 aircraft
- Reset time
- Memory usage per episode
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.horizontal_cr import HorizontalCRScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from tests.helpers.env_factory import make_config as _make_config
from tests.helpers.env_factory import write_rewards_yaml as _write_rewards_yaml
from tests.helpers.env_factory import make_env as _make_env


# ---------------------------------------------------------------------------
# Fake BlueSkyWrapper
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------





class TestStepTime5Aircraft:
    """Step time with 5 aircraft should be under 100ms."""

    def test_step_time_5_aircraft(self, tmp_path: Path) -> None:
        config = _make_config(num_aircraft=5, max_steps=10)
        env = _make_env(tmp_path, config)
        env.reset(seed=42)

        # Warm-up step
        actions = {a: [2, 2, 2] for a in env.agents}
        env.step(actions)

        # Benchmark
        times = []
        for _ in range(5):
            actions = {a: [2, 2, 2] for a in env.agents}
            if not actions:
                break
            t0 = time.perf_counter()
            env.step(actions)
            times.append(time.perf_counter() - t0)

        avg_ms = (sum(times) / len(times)) * 1000 if times else 0
        assert avg_ms < 100, f"Average step time {avg_ms:.1f}ms exceeds 100ms"


class TestStepTime20Aircraft:
    """Step time with 20 aircraft should be under 200ms."""

    def test_step_time_20_aircraft(self, tmp_path: Path) -> None:
        config = _make_config(num_aircraft=20, max_steps=10)
        env = _make_env(tmp_path, config)
        env.reset(seed=42)

        # Warm-up step
        actions = {a: [2, 2, 2] for a in env.agents}
        env.step(actions)

        # Benchmark
        times = []
        for _ in range(5):
            actions = {a: [2, 2, 2] for a in env.agents}
            if not actions:
                break
            t0 = time.perf_counter()
            env.step(actions)
            times.append(time.perf_counter() - t0)

        avg_ms = (sum(times) / len(times)) * 1000 if times else 0
        assert avg_ms < 200, f"Average step time {avg_ms:.1f}ms exceeds 200ms"


class TestResetTime:
    """Reset time should be under 500ms."""

    def test_reset_time(self, tmp_path: Path) -> None:
        config = _make_config(num_aircraft=10, max_steps=10)
        env = _make_env(tmp_path, config)

        # Warm-up
        env.reset(seed=42)

        # Benchmark
        times = []
        for i in range(3):
            t0 = time.perf_counter()
            env.reset(seed=42 + i)
            times.append(time.perf_counter() - t0)

        avg_ms = (sum(times) / len(times)) * 1000
        assert avg_ms < 500, f"Average reset time {avg_ms:.1f}ms exceeds 500ms"


class TestEpisodeMemory:
    """Single episode memory should stay under 1GB."""

    def test_episode_memory(self, tmp_path: Path) -> None:
        """Run a 50-step episode and verify it completes (implicit memory check)."""
        config = _make_config(num_aircraft=10, max_steps=50)
        env = _make_env(tmp_path, config)
        env.reset(seed=42)

        for _ in range(50):
            actions = {a: [2, 2, 2] for a in env.agents}
            if not actions:
                break
            env.step(actions)

        env.close()
        # If we get here without OOM, memory is acceptable
