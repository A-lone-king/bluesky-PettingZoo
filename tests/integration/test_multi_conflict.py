"""Integration tests — multi-conflict scenario (T-16).

Five aircraft in crossing paths. Validates:
  1. Multiple conflict pairs detected simultaneously
  2. Aircraft entering/leaving airspace lifecycle
  3. infos contains all required fields
  4. Episode can complete without crash
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

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from tests.helpers.env_factory import make_config as _make_config
from tests.helpers.env_factory import _DEFAULT_REWARDS as _make_rewards_config
from tests.helpers.env_factory import make_env as _make_env


# ---------------------------------------------------------------------------
# Fake BlueSkyWrapper — in-memory, no real BlueSky dependency
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Config & env factory
# ---------------------------------------------------------------------------





def _place_aircraft(
    env: BlueSkyMARLEnv,
    positions: dict[str, dict[str, float]],
) -> None:
    """Manually place aircraft at specific positions after reset."""
    wrapper = env._wrapper
    for acid, pos in positions.items():
        if acid in wrapper.get_active_aircraft_ids():
            wrapper.set_aircraft_state(acid, **pos)


# ===========================================================================
# T-16 Test Cases
# ===========================================================================


class TestMultiConflictDetection:
    """Five aircraft crossing paths — multiple conflict pairs detected simultaneously."""

    def test_multi_conflict_detection(self) -> None:
        """5 aircraft in a cluster: multiple conflict pairs detected."""
        env = _make_env(initial_count=5, max_steps=3)
        env.reset(seed=42)

        # Place 5 aircraft close together (~3-7NM apart) at same altitude
        # AC000-AC001: ~3NM (warning/NMAC zone)
        # AC000-AC002: ~6NM (warning zone)
        # AC001-AC003: ~8NM (warning zone)
        # AC004: ~12NM from center (safe but within perception)
        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 33000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.25, "lon": 116.30, "alt": 33000, "hdg": 270, "tas": 450},
            "AC002": {"lat": 39.30, "lon": 116.25, "alt": 33000, "hdg": 180, "tas": 450},
            "AC003": {"lat": 39.30, "lon": 116.33, "alt": 33000, "hdg": 0, "tas": 450},
            "AC004": {"lat": 39.10, "lon": 116.50, "alt": 36000, "hdg": 45, "tas": 450},
        })

        actions = {a: [2, 2, 2] for a in env.agents}
        _, rewards, terms, truncs, infos = env.step(actions)

        # At least 2 agents should have conflict (in infos, including terminated ones)
        conflict_agents = [
            aid for aid in infos
            if infos[aid].get("textual_state", {}).get("conflict_status") in ("warning", "nmac")
        ]
        assert len(conflict_agents) >= 2, (
            f"Expected at least 2 conflict agents, got {len(conflict_agents)}: "
            f"statuses={ {a: infos.get(a, {}).get('textual_state', {}).get('conflict_status') for a in infos} }"
        )

        # AC004 should be safe (far from the cluster)
        if "AC004" in infos:
            status_004 = infos["AC004"]["textual_state"]["conflict_status"]
            assert status_004 == "safe", f"AC004 should be safe, got {status_004}"


class TestAgentLifecycle:
    """Aircraft entering/leaving airspace — full lifecycle."""

    def test_agent_lifecycle(self) -> None:
        """Aircraft leaving airspace should be removed from agents list."""
        env = _make_env(initial_count=3, max_steps=30)
        # Use smaller bounds so AC002 can actually depart
        env._airspace = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 117.0}
        env._wrapper._airspace_bounds = env._airspace
        env.reset(seed=42)

        # Place 2 aircraft far apart (stable), 1 near boundary heading out
        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 33000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.25, "lon": 116.75, "alt": 33000, "hdg": 90, "tas": 450},
            "AC002": {"lat": 39.25, "lon": 116.95, "alt": 33000, "hdg": 90, "tas": 450},
        })

        initial_agents = set(env.agents)
        assert len(initial_agents) == 3

        departed = set()
        for step_i in range(30):
            if not env.agents:
                break
            actions = {a: [2, 2, 2] for a in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)

            # Track agents that have been removed
            current_agents = set(env.agents)
            departed |= initial_agents - current_agents

            # Terminations should match removed agents
            for aid in terms:
                if aid not in current_agents:
                    assert terms[aid] is True, (
                        f"Step {step_i}: {aid} removed but termination={terms[aid]}"
                    )

        # AC002 should have departed (near eastern boundary, heading east)
        assert "AC002" in departed, (
            f"AC002 should have departed. Departed: {departed}, remaining: {set(env.agents)}"
        )


class TestInfosCompleteness:
    """infos dict should contain all required fields for all agents."""

    def test_infos_completeness(self) -> None:
        """infos contains textual_state and airspace_snapshot for every agent."""
        env = _make_env(initial_count=5, max_steps=3)
        env.reset(seed=42)

        # Place aircraft spread out to avoid NMAC
        _place_aircraft(env, {
            "AC000": {"lat": 39.10, "lon": 116.10, "alt": 30000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.15, "lon": 116.30, "alt": 31000, "hdg": 90, "tas": 450},
            "AC002": {"lat": 39.20, "lon": 116.50, "alt": 32000, "hdg": 90, "tas": 450},
            "AC003": {"lat": 39.30, "lon": 116.20, "alt": 34000, "hdg": 90, "tas": 450},
            "AC004": {"lat": 39.40, "lon": 116.40, "alt": 36000, "hdg": 90, "tas": 450},
        })

        actions = {a: [2, 2, 2] for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)

        # All agents in agents_snapshot should have infos
        for agent_id in obs:
            assert agent_id in infos, f"{agent_id} missing from infos"

        # Check required fields
        for agent_id in infos:
            info = infos[agent_id]
            assert "textual_state" in info, f"{agent_id} missing textual_state"
            assert "airspace_snapshot" in info, f"{agent_id} missing airspace_snapshot"

            ts = info["textual_state"]
            assert "conflict_status" in ts, f"{agent_id} textual_state missing conflict_status"
            assert isinstance(ts["conflict_status"], str)
            assert ts["conflict_status"] in ("safe", "warning", "nmac")

            snap = info["airspace_snapshot"]
            assert isinstance(snap, dict), f"{agent_id} airspace_snapshot not a dict"


class TestEpisodeCompletion:
    """5-aircraft episode can complete without crash."""

    def test_episode_completion(self) -> None:
        """Run a full episode with 5 aircraft — completes without exception."""
        max_steps = 30
        env = _make_env(initial_count=5, max_steps=max_steps)
        env.reset(seed=42)

        # Place aircraft on parallel tracks — safe spacing
        _place_aircraft(env, {
            "AC000": {"lat": 39.10, "lon": 116.10, "alt": 30000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.15, "lon": 116.20, "alt": 31000, "hdg": 90, "tas": 450},
            "AC002": {"lat": 39.20, "lon": 116.30, "alt": 32000, "hdg": 90, "tas": 450},
            "AC003": {"lat": 39.30, "lon": 116.40, "alt": 34000, "hdg": 90, "tas": 450},
            "AC004": {"lat": 39.40, "lon": 116.50, "alt": 36000, "hdg": 90, "tas": 450},
        })

        total_reward = 0.0
        step_count = 0
        for _ in range(max_steps):
            if not env.agents:
                break
            actions = {a: [2, 2, 2] for a in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)
            total_reward += sum(rewards.values())
            step_count += 1

            if all(truncs.values()):
                break

        # Episode ran at least 1 step
        assert step_count >= 1
        # Total reward should be finite
        assert math.isfinite(total_reward), f"Total reward not finite: {total_reward}"

    def test_episode_with_actions(self) -> None:
        """Episode with mixed actions (some turning, some straight) completes."""
        max_steps = 20
        env = _make_env(initial_count=5, max_steps=max_steps)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.10, "lon": 116.10, "alt": 30000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.15, "lon": 116.20, "alt": 31000, "hdg": 90, "tas": 450},
            "AC002": {"lat": 39.20, "lon": 116.30, "alt": 32000, "hdg": 90, "tas": 450},
            "AC003": {"lat": 39.30, "lon": 116.40, "alt": 34000, "hdg": 90, "tas": 450},
            "AC004": {"lat": 39.40, "lon": 116.50, "alt": 36000, "hdg": 90, "tas": 450},
        })

        step_count = 0
        for _ in range(max_steps):
            if not env.agents:
                break
            # Alternate: some agents turn, others go straight
            actions = {}
            for i, a in enumerate(env.agents):
                if i % 2 == 0:
                    actions[a] = [4, 2, 2]  # heading +20
                else:
                    actions[a] = [2, 2, 2]  # no change
            obs, rewards, terms, truncs, infos = env.step(actions)
            step_count += 1

            if all(truncs.values()):
                break

        assert step_count >= 1
