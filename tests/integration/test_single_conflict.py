"""Integration tests — single-conflict scenario (T-15).

Two aircraft on converging courses. Validates:
  1. Conflict is correctly detected via reward and textual state
  2. NMAC triggers agent termination
  3. Conflict penalty is correctly applied
  4. Reward component weights are correctly applied
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty

from tests.helpers.fake_wrapper import FakeBlueSkyWrapper
from tests.helpers.env_factory import make_config as _make_config
from tests.helpers.env_factory import _DEFAULT_REWARDS as _make_rewards_config
from tests.helpers.env_factory import make_env as _make_env


# ---------------------------------------------------------------------------
# Fake BlueSkyWrapper
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Config & env factory
# ---------------------------------------------------------------------------





def _place_aircraft(
    env: BlueSkyMARLEnv,
    positions: dict[str, dict[str, float]],
) -> None:
    wrapper = env._wrapper
    for acid, pos in positions.items():
        if acid in wrapper._aircraft:
            wrapper._aircraft[acid].update(pos)


# ===========================================================================
# T-15 Test Cases
# ===========================================================================


class TestConflictDetected:
    """Two aircraft in close proximity should be detected as conflicting."""

    def test_conflict_detected(self) -> None:
        """Aircraft <10NM apart, same altitude: conflict detected via reward and status."""
        env = _make_env(initial_count=2, max_steps=2)
        env.reset(seed=42)

        # ~3NM apart, same altitude → warning-level conflict
        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.25, "lon": 116.30, "alt": 35000, "hdg": 270, "tas": 450},
        })

        actions = {a: [2, 2, 2] for a in env.agents}
        _, rewards, _, _, infos = env.step(actions)

        # Both agents should receive conflict penalty
        for agent_id in env.agents:
            assert rewards[agent_id] < 0, (
                f"{agent_id} expected conflict penalty, got {rewards[agent_id]}"
            )

        # Textual state should report conflict
        for agent_id in infos:
            status = infos[agent_id]["textual_state"]["conflict_status"]
            assert status in ("warning", "nmac"), (
                f"{agent_id} expected conflict status, got {status}"
            )


class TestNMACTriggersTermination:
    """NMAC (Near Mid-Air Collision) should terminate the involved agents."""

    def test_nmac_triggers_termination(self) -> None:
        """Aircraft <5NM apart, same altitude, same heading toward each other: NMAC."""
        env = _make_env(initial_count=2, max_steps=5)
        env.reset(seed=42)

        # ~1NM apart, same altitude, heading toward each other
        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.25, "lon": 116.27, "alt": 35000, "hdg": 270, "tas": 450},
        })

        actions = {a: [2, 2, 2] for a in env.agents}
        _, rewards, terminations, _, infos = env.step(actions)

        # NMAC should be detected
        for agent_id in infos:
            assert infos[agent_id]["textual_state"]["conflict_status"] == "nmac"

        # NMAC penalty should be applied
        for agent_id in rewards:
            assert rewards[agent_id] <= -100.0, (
                f"{agent_id} expected NMAC penalty, got {rewards[agent_id]}"
            )

        # NMAC should trigger termination for involved agents
        for agent_id in terminations:
            assert terminations[agent_id] is True, (
                f"{agent_id} should be terminated after NMAC"
            )


class TestConflictPenaltyApplied:
    """Verify conflict penalty values match the configuration."""

    def test_conflict_penalty_applied(self) -> None:
        """Verify NMAC, warning, and separation penalties are correctly applied."""
        env = _make_env(initial_count=2, max_steps=2)

        # --- NMAC case ---
        env.reset(seed=42)
        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.25, "lon": 116.26, "alt": 35000, "hdg": 270, "tas": 450},
        })
        actions = {a: [2, 2, 2] for a in env.agents}
        _, rewards_nmac, _, _, _ = env.step(actions)
        assert rewards_nmac["AC000"] <= -100.0

        # --- Warning case ---
        # ~7.8NM apart (0.13 deg lat), same altitude → warning but not NMAC
        env2 = _make_env(initial_count=2, max_steps=2)
        env2.reset(seed=42)
        _place_aircraft(env2, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.38, "lon": 116.25, "alt": 35000, "hdg": 270, "tas": 450},
        })
        actions2 = {a: [2, 2, 2] for a in env2.agents}
        _, rewards_warn, _, _, _ = env2.step(actions2)
        # Warning penalty (-10) * weight (1.0) + other components
        assert rewards_warn["AC000"] < 0
        assert rewards_warn["AC000"] > -100.0  # Not NMAC level

        # --- Safe case ---
        env3 = _make_env(initial_count=2, max_steps=2)
        env3.reset(seed=42)
        _place_aircraft(env3, {
            "AC000": {"lat": 39.10, "lon": 116.10, "alt": 30000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.40, "lon": 116.40, "alt": 36000, "hdg": 270, "tas": 450},
        })
        actions3 = {a: [2, 2, 2] for a in env3.agents}
        _, rewards_safe, _, _, _ = env3.step(actions3)
        # No conflict penalty, only step penalty
        assert rewards_safe["AC000"] > -1.0


class TestRewardComponentWeights:
    """Verify that reward component weights are correctly applied."""

    def test_reward_component_weights(self) -> None:
        """Verify that conflict (1.0), smoothness (0.5), and efficiency (0.3) weights apply."""
        env = _make_env(initial_count=2, max_steps=2)
        env.reset(seed=42)

        # Aircraft in warning zone (~7.8NM), no action → no smoothness penalty
        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.38, "lon": 116.25, "alt": 35000, "hdg": 270, "tas": 450},
        })

        # No-op action: conflict + efficiency only (no smoothness)
        actions_noop = {a: [2, 2, 2] for a in env.agents}
        _, rewards_noop, _, _, _ = env.step(actions_noop)

        # Action with heading change: conflict + efficiency + smoothness
        env2 = _make_env(initial_count=2, max_steps=2)
        env2.reset(seed=42)
        _place_aircraft(env2, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.38, "lon": 116.25, "alt": 35000, "hdg": 270, "tas": 450},
        })
        actions_turn = {"AC000": [4, 2, 2], "AC001": [2, 2, 2]}
        _, rewards_turn, _, _, _ = env2.step(actions_turn)

        # Turning action adds smoothness penalty (-0.1 * 0.5 = -0.05)
        assert rewards_turn["AC000"] < rewards_noop["AC000"], (
            "Turning action should have lower reward due to smoothness penalty"
        )

        # Verify the difference is approximately the smoothness penalty
        diff = rewards_noop["AC000"] - rewards_turn["AC000"]
        assert diff == pytest.approx(0.05, rel=0.1), (
            f"Expected smoothness penalty ~0.05, got {diff}"
        )
