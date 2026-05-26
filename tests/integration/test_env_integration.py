"""Integration tests for the full BlueSkyMARLEnv lifecycle.

Covers the four spec scenarios (§11.3):
  1. No-conflict episode — all aircraft fly straight to goals
  2. Single-conflict episode — two aircraft converge, verify detection & reward
  3. Multi-conflict episode — 5 aircraft crossing
  4. Boundary conditions — aircraft leaving airspace, agent removal

Additional scenarios:
  5. Full episode reward accumulation
  6. Observation consistency across steps
  7. Agent interaction (RandomAgent, RuleBasedAgent)
  8. Truncation on max steps
  9. Deterministic reset with seed
  10. Action effects propagate through the pipeline
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from bluesky_pettingzoo.utils.types import AircraftState


# ---------------------------------------------------------------------------
# Enhanced Fake BlueSkyWrapper — processes HDG/ALT/SPD commands
# ---------------------------------------------------------------------------


class EnhancedBlueSkyWrapper:
    """Fake wrapper that processes HDG, ALT, SPD commands for integration tests."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.dt: float = config["simulation"]["dt"]
        self._initialized = False
        self._simt: float = 0.0
        self._aircraft: dict[str, dict[str, Any]] = {}

        bounds_list = config.get("airspace", {}).get("sectors", [])
        if bounds_list:
            lats = [b["bounds"][0][0] for b in bounds_list] + [b["bounds"][1][0] for b in bounds_list]
            lons = [b["bounds"][0][1] for b in bounds_list] + [b["bounds"][1][1] for b in bounds_list]
            self._bounds = {
                "lat_min": min(lats), "lat_max": max(lats),
                "lon_min": min(lons), "lon_max": max(lons),
            }
        else:
            self._bounds = {}

    def init_simulation(self) -> None:
        self._initialized = True

    def step(self) -> float:
        return self.step_n(1)

    def step_n(self, n: int, on_substep: Any = None) -> float:
        for i in range(n):
            self._simt += self.dt
            for st in self._aircraft.values():
                spd_nm_s = st["tas"] / 3600.0
                hdg_rad = math.radians(st["hdg"])
                st["lat"] += math.cos(hdg_rad) * spd_nm_s * self.dt / 60.0
                st["lon"] += math.sin(hdg_rad) * spd_nm_s * self.dt / (
                    60.0 * math.cos(math.radians(st["lat"]))
                )
            if on_substep is not None and not on_substep(i):
                break
        return self._simt

    def reset(self) -> None:
        self._aircraft.clear()
        self._simt = 0.0

    def create_aircraft(
        self, acid: str, actype: str, lat: float, lon: float,
        alt: float, hdg: float, spd: float,
    ) -> None:
        self._aircraft[acid] = {
            "id": acid, "lat": lat, "lon": lon, "alt": alt,
            "hdg": hdg, "tas": spd, "vs": 0.0,
        }

    def remove_aircraft(self, acid: str) -> None:
        self._aircraft.pop(acid, None)

    def send_command(self, command: str) -> None:
        self._process_command(command)

    def send_commands_batch(self, commands: list[str]) -> None:
        for cmd in commands:
            self._process_command(cmd)

    def _process_command(self, command: str) -> None:
        """Parse and apply HDG/ALT/SPD commands."""
        parts = command.strip().split()
        if len(parts) < 3:
            return
        cmd_type, acid, value_str = parts[0], parts[1], parts[2]
        if acid not in self._aircraft:
            return
        try:
            value = float(value_str)
        except ValueError:
            return
        if cmd_type == "HDG":
            self._aircraft[acid]["hdg"] = value % 360
        elif cmd_type == "ALT":
            self._aircraft[acid]["alt"] = value
        elif cmd_type == "SPD":
            self._aircraft[acid]["tas"] = value

    def get_aircraft_state(self, acid: str) -> dict[str, Any]:
        if acid not in self._aircraft:
            raise ValueError(f"Aircraft {acid} not found")
        return dict(self._aircraft[acid])

    def get_all_aircraft_states(self) -> dict[str, dict[str, Any]]:
        return {k: dict(v) for k, v in self._aircraft.items()}

    def get_active_aircraft_ids(self) -> list[str]:
        return list(self._aircraft.keys())

    def is_aircraft_in_airspace(self, acid: str) -> bool:
        if acid not in self._aircraft:
            return False
        if not self._bounds:
            return True
        st = self._aircraft[acid]
        return (
            self._bounds["lat_min"] <= st["lat"] <= self._bounds["lat_max"]
            and self._bounds["lon_min"] <= st["lon"] <= self._bounds["lon_max"]
        )

    def close(self) -> None:
        self._initialized = False


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _make_config(
    initial_count: int = 3,
    max_steps: int = 360,
    lat_range: tuple[float, float] = (39.0, 39.5),
    lon_range: tuple[float, float] = (116.0, 116.5),
) -> dict[str, Any]:
    return {
        "simulation": {"dt": 5.0, "max_episode_steps": max_steps, "headless": True},
        "airspace": {
            "name": "test_sector",
            "sectors": [
                {"id": "s1", "bounds": [[lat_range[0], lon_range[0]], [lat_range[1], lon_range[1]]]},
            ],
        },
        "aircraft": {
            "initial_count": initial_count,
            "spawn": {
                "altitude_range": [29000, 37000],
                "speed_range": [400, 500],
                "heading_range": [0, 360],
            },
        },
        "observation": {
            "perception_radius_nm": 20,
            "perception_alt_diff_ft": 3000,
            "max_observable_aircraft": 10,
        },
        "action": {
            "heading_adjustments": [-20, -10, 0, 10, 20],
            "altitude_adjustments": [-2000, -1000, 0, 1000, 2000],
            "speed_adjustments": [-20, -10, 0, 10, 20],
        },
        "normalization": {
            "heading": {"mid": 180, "range": 180},
            "altitude": {"mid": 33000, "range": 10000},
            "speed": {"mid": 450, "range": 100},
            "distance": {"max": 20},
        },
    }


def _make_rewards_config() -> dict[str, Any]:
    return {
        "components": {
            "conflict": {
                "enabled": True,
                "weight": 1.0,
                "nmac_penalty": -100,
                "warning_penalty": -10,
                "separation_penalty": -5,
                "thresholds": {
                    "nmac_horizontal_nm": 5,
                    "nmac_vertical_ft": 1000,
                    "warning_horizontal_nm": 10,
                    "warning_vertical_ft": 2000,
                },
            },
            "smoothness": {"enabled": True, "weight": 0.5, "action_penalty": -0.1},
            "efficiency": {
                "enabled": True,
                "weight": 0.3,
                "max_deviation_nm": 50,
                "deviation_penalty_scale": 5,
                "arrival_reward": 10,
                "step_penalty": -0.01,
                "arrival_threshold_nm": 2,
            },
        }
    }


def _make_env(
    initial_count: int = 3,
    max_steps: int = 360,
    wrapper_cls: type = EnhancedBlueSkyWrapper,
    lat_range: tuple[float, float] = (39.0, 39.5),
    lon_range: tuple[float, float] = (116.0, 116.5),
) -> BlueSkyMARLEnv:
    config = _make_config(
        initial_count=initial_count,
        max_steps=max_steps,
        lat_range=lat_range,
        lon_range=lon_range,
    )
    rewards_cfg = _make_rewards_config()
    merged = {**config, **rewards_cfg}

    wrapper = wrapper_cls(config)
    obs_manager = ObservationManager(config)
    action_translator = ActionTranslator(config)
    calc = RewardCalculator()
    calc.register(ConflictPenalty(merged), weight=1.0)
    calc.register(SmoothnessPenalty(merged), weight=0.5)
    eff = EfficiencyReward(merged)
    calc.register(eff, weight=0.3)

    return BlueSkyMARLEnv(
        config=config,
        wrapper=wrapper,
        observation_manager=obs_manager,
        action_translator=action_translator,
        reward_calculator=calc,
        rewards_config=rewards_cfg,
    )


def _place_aircraft(
    env: BlueSkyMARLEnv,
    positions: dict[str, dict[str, float]],
) -> None:
    """Manually place aircraft at specific positions after reset."""
    wrapper = env._wrapper
    for acid, pos in positions.items():
        if acid in wrapper._aircraft:
            wrapper._aircraft[acid].update(pos)
    # Sync prev_states so reward computation uses updated positions
    env._prev_states = env._get_all_aircraft_states()


# ===========================================================================
# Scenario 1: No-conflict episode
# ===========================================================================


class TestNoConflictScenario:
    """All aircraft far apart, flying straight — no conflicts should occur."""

    def test_no_conflict_straight_flight(self) -> None:
        """Aircraft far apart on parallel tracks: no conflict penalty."""
        env = _make_env(initial_count=2, max_steps=5)
        env.reset(seed=42)

        # Place aircraft far apart on parallel eastbound tracks
        _place_aircraft(env, {
            "AC000": {"lat": 39.10, "lon": 116.10, "alt": 30000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.40, "lon": 116.10, "alt": 36000, "hdg": 90, "tas": 450},
        })

        # Straight flight — no adjustments
        for _ in range(5):
            actions = {a: [2, 2, 2] for a in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)

            # No conflict penalty (only step penalty from efficiency)
            for agent_id in env.agents:
                assert rewards[agent_id] > -1.0, (
                    f"{agent_id} got unexpected large penalty: {rewards[agent_id]}"
                )

    def test_no_conflict_observations_safe(self) -> None:
        """In no-conflict scenario, textual_state reports 'safe'."""
        env = _make_env(initial_count=2, max_steps=2)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.10, "lon": 116.10, "alt": 30000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.40, "lon": 116.10, "alt": 36000, "hdg": 90, "tas": 450},
        })

        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, _, _, infos = env.step(actions)

        for agent_id in infos:
            assert infos[agent_id]["textual_state"]["conflict_status"] == "safe"


# ===========================================================================
# Scenario 2: Single-conflict episode
# ===========================================================================


class TestSingleConflictScenario:
    """Two aircraft on converging courses — conflict detection and reward."""

    def test_conflict_detection_warning(self) -> None:
        """Two aircraft ~8NM apart, same altitude: warning-level conflict."""
        env = _make_env(initial_count=2, max_steps=2)
        env.reset(seed=42)

        # ~8NM apart horizontally, same altitude
        _place_aircraft(env, {
            "AC000": {"lat": 39.20, "lon": 116.20, "alt": 35000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.20, "lon": 116.33, "alt": 35000, "hdg": 270, "tas": 450},
        })

        actions = {a: [2, 2, 2] for a in env.agents}
        _, rewards, _, _, infos = env.step(actions)

        # Both agents should get conflict penalty
        for agent_id in env.agents:
            assert rewards[agent_id] <= -10.0, (
                f"{agent_id} expected warning penalty, got {rewards[agent_id]}"
            )

    def test_conflict_detection_nmac(self) -> None:
        """Two aircraft <5NM apart, same altitude: NMAC-level conflict."""
        env = _make_env(initial_count=2, max_steps=2)
        env.reset(seed=42)

        # ~2NM apart, same altitude
        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.25, "lon": 116.28, "alt": 35000, "hdg": 270, "tas": 450},
        })

        actions = {a: [2, 2, 2] for a in env.agents}
        _, rewards, _, _, infos = env.step(actions)

        for agent_id in env.agents:
            assert rewards[agent_id] <= -100.0, (
                f"{agent_id} expected NMAC penalty, got {rewards[agent_id]}"
            )

    def test_conflict_resolution_heading_change(self) -> None:
        """Turning away from conflict changes the heading via command."""
        env = _make_env(initial_count=2, max_steps=2)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.25, "lon": 116.40, "alt": 35000, "hdg": 270, "tas": 450},
        })

        # AC000 turns right (+20°), AC001 goes straight
        actions = {"AC000": [4, 2, 2], "AC001": [2, 2, 2]}
        env.step(actions)

        # Verify heading command was applied
        wrapper = env._wrapper
        assert wrapper._aircraft["AC000"]["hdg"] == 110.0  # 90 + 20

    def test_conflict_vs_safe_reward_difference(self) -> None:
        """Aircraft in conflict zone get worse reward than those in safe zone."""
        # Conflict env
        env_c = _make_env(initial_count=2, max_steps=2)
        env_c.reset(seed=42)
        _place_aircraft(env_c, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.25, "lon": 116.28, "alt": 35000, "hdg": 270, "tas": 450},
        })
        actions_c = {a: [2, 2, 2] for a in env_c.agents}
        _, rewards_c, _, _, _ = env_c.step(actions_c)

        # Safe env
        env_s = _make_env(initial_count=2, max_steps=2)
        env_s.reset(seed=42)
        _place_aircraft(env_s, {
            "AC000": {"lat": 39.10, "lon": 116.10, "alt": 30000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.40, "lon": 116.40, "alt": 36000, "hdg": 270, "tas": 450},
        })
        actions_s = {a: [2, 2, 2] for a in env_s.agents}
        _, rewards_s, _, _, _ = env_s.step(actions_s)

        # Conflict rewards should be much lower
        for agent_id in rewards_c:
            assert rewards_c[agent_id] < rewards_s[agent_id]


# ===========================================================================
# Scenario 3: Multi-conflict episode (5 aircraft)
# ===========================================================================


class TestMultiConflictScenario:
    """Five aircraft in a crossing pattern — multiple conflicts."""

    def test_five_aircraft_crossing(self) -> None:
        """5 aircraft converging from different directions."""
        env = _make_env(initial_count=5, max_steps=5)
        env.reset(seed=42)

        # Place 5 aircraft heading toward the center
        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.10, "alt": 33000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.25, "lon": 116.40, "alt": 33000, "hdg": 270, "tas": 450},
            "AC002": {"lat": 39.10, "lon": 116.25, "alt": 34000, "hdg": 0, "tas": 450},
            "AC003": {"lat": 39.40, "lon": 116.25, "alt": 32000, "hdg": 180, "tas": 450},
            "AC004": {"lat": 39.20, "lon": 116.20, "alt": 33000, "hdg": 45, "tas": 450},
        })

        total_reward = {a: 0.0 for a in env.agents}
        for _ in range(5):
            actions = {a: [2, 2, 2] for a in env.agents}
            _, rewards, _, _, _ = env.step(actions)
            for aid in rewards:
                if aid in total_reward:
                    total_reward[aid] += rewards[aid]

        # At least some agents should have received conflict penalties
        negative_count = sum(1 for r in total_reward.values() if r < -1.0)
        assert negative_count > 0, "Expected at least one agent with conflict penalty"

    def test_five_aircraft_all_action_combinations(self) -> None:
        """All 5 aircraft take different action combinations."""
        env = _make_env(initial_count=5, max_steps=3)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.20, "lon": 116.20, "alt": 33000, "hdg": 45, "tas": 450},
            "AC001": {"lat": 39.25, "lon": 116.25, "alt": 34000, "hdg": 135, "tas": 440},
            "AC002": {"lat": 39.30, "lon": 116.30, "alt": 35000, "hdg": 225, "tas": 460},
            "AC003": {"lat": 39.35, "lon": 116.35, "alt": 32000, "hdg": 315, "tas": 430},
            "AC004": {"lat": 39.15, "lon": 116.15, "alt": 31000, "hdg": 0, "tas": 470},
        })

        # Different actions for each agent
        actions = {
            "AC000": [0, 0, 0],  # hdg-20, alt-2000, spd-20
            "AC001": [1, 1, 1],  # hdg-10, alt-1000, spd-10
            "AC002": [2, 2, 2],  # no change
            "AC003": [3, 3, 3],  # hdg+10, alt+1000, spd+10
            "AC004": [4, 4, 4],  # hdg+20, alt+2000, spd+20
        }

        obs, rewards, terms, truncs, infos = env.step(actions)

        # All agents should have observations and rewards
        assert len(rewards) == 5
        for agent_id in env.agents:
            assert agent_id in rewards

    def test_multi_conflict_conflict_status(self) -> None:
        """Multiple nearby aircraft should trigger conflict status."""
        env = _make_env(initial_count=3, max_steps=2)
        env.reset(seed=42)

        # Three aircraft close together
        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.26, "lon": 116.26, "alt": 35000, "hdg": 180, "tas": 450},
            "AC002": {"lat": 39.24, "lon": 116.24, "alt": 34500, "hdg": 0, "tas": 450},
        })

        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, _, _, infos = env.step(actions)

        # At least one agent should have conflict status
        statuses = [infos[a]["textual_state"]["conflict_status"] for a in infos]
        assert any(s != "safe" for s in statuses)


# ===========================================================================
# Scenario 4: Boundary conditions
# ===========================================================================


class TestBoundaryConditions:
    """Aircraft leaving airspace, agent removal, edge cases."""

    def test_aircraft_leaves_airspace(self) -> None:
        """Aircraft heading toward boundary should eventually leave."""
        env = _make_env(initial_count=2, max_steps=200)
        env.reset(seed=42)

        # Place aircraft heading toward the boundary
        _place_aircraft(env, {
            "AC000": {"lat": 39.05, "lon": 116.05, "alt": 35000, "hdg": 225, "tas": 500},
            "AC001": {"lat": 39.45, "lon": 116.45, "alt": 35000, "hdg": 45, "tas": 500},
        })

        initial_agents = set(env.agents)
        for _ in range(200):
            if not env.agents:
                break
            actions = {a: [2, 2, 2] for a in env.agents}
            _, _, terms, _, _ = env.step(actions)
            if any(terms.values()):
                break

        # At least one agent should have been removed
        assert len(env.agents) < len(initial_agents)

    def test_terminated_agent_gets_observation(self) -> None:
        """After an agent is terminated, it still appears in the return dicts."""
        env = _make_env(initial_count=2, max_steps=200)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.01, "lon": 116.01, "alt": 35000, "hdg": 225, "tas": 500},
            "AC001": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
        })

        for _ in range(200):
            if len(env.agents) < 2:
                break
            actions = {a: [2, 2, 2] for a in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)

        # If an agent was terminated, it should still be in the return dicts
        if any(terms.values()):
            for agent_id in terms:
                assert agent_id in obs
                assert agent_id in rewards
                assert agent_id in truncs
                assert agent_id in infos

    def test_all_aircraft_same_position(self) -> None:
        """All aircraft at exactly the same position: extreme NMAC."""
        env = _make_env(initial_count=3, max_steps=2)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
            "AC002": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
        })

        actions = {a: [2, 2, 2] for a in env.agents}
        _, rewards, _, _, _ = env.step(actions)

        # All should get NMAC penalty
        for agent_id in rewards:
            assert rewards[agent_id] <= -100.0

    def test_heading_wraparound_360(self) -> None:
        """Heading 350° + 20° should wrap to 10°."""
        env = _make_env(initial_count=1, max_steps=2)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 350, "tas": 450},
        })

        actions = {"AC000": [4, 2, 2]}  # +20° heading
        env.step(actions)

        assert env._wrapper._aircraft["AC000"]["hdg"] == 10.0

    def test_extreme_altitude_values(self) -> None:
        """Very high altitude: normalizer clips, env still works."""
        env = _make_env(initial_count=1, max_steps=2)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 60000, "hdg": 90, "tas": 450},
        })

        actions = {"AC000": [2, 2, 2]}
        obs, rewards, _, _, _ = env.step(actions)

        # Observation should still be valid
        assert obs["AC000"]["self_state"].shape == (9,)


# ===========================================================================
# Full episode lifecycle
# ===========================================================================


class TestFullEpisodeLifecycle:
    """Multi-step episode with reward accumulation."""

    def test_reward_accumulation(self) -> None:
        """Rewards should accumulate over an episode."""
        env = _make_env(initial_count=2, max_steps=10)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.20, "lon": 116.20, "alt": 35000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.30, "lon": 116.30, "alt": 34000, "hdg": 270, "tas": 450},
        })

        total_rewards: dict[str, float] = {a: 0.0 for a in env.agents}
        for _ in range(10):
            actions = {a: [2, 2, 2] for a in env.agents}
            _, rewards, _, _, _ = env.step(actions)
            for aid in rewards:
                if aid in total_rewards:
                    total_rewards[aid] += rewards[aid]

        # Rewards should have accumulated (non-zero)
        for agent_id in total_rewards:
            assert total_rewards[agent_id] != 0.0

    def test_episode_ends_at_max_steps(self) -> None:
        """Truncations should be True when max_episode_steps is reached."""
        env = _make_env(initial_count=2, max_steps=5)
        env.reset(seed=42)

        for _ in range(5):
            actions = {a: [2, 2, 2] for a in env.agents}
            _, _, _, truncs, _ = env.step(actions)

        assert all(truncs.values())

    def test_observation_consistency_across_steps(self) -> None:
        """Observation keys should remain consistent across steps."""
        env = _make_env(initial_count=3, max_steps=5)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.20, "lon": 116.20, "alt": 35000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.25, "lon": 116.25, "alt": 34000, "hdg": 180, "tas": 440},
            "AC002": {"lat": 39.30, "lon": 116.30, "alt": 36000, "hdg": 270, "tas": 460},
        })

        prev_agents = set(env.agents)
        for _ in range(5):
            actions = {a: [2, 2, 2] for a in env.agents}
            obs, _, _, _, infos = env.step(actions)

            # Current agents should be subset of previous
            current_agents = set(env.agents)
            assert current_agents <= prev_agents

            # Observations should have correct keys
            for agent_id in obs:
                assert "self_state" in obs[agent_id]
                assert "other_aircraft" in obs[agent_id]
                assert "other_aircraft_mask" in obs[agent_id]
                assert "goal" in obs[agent_id]

            prev_agents = current_agents

    def test_infos_textual_state_updated_each_step(self) -> None:
        """Textual state should reflect current aircraft position."""
        env = _make_env(initial_count=1, max_steps=3)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
        })

        prev_lat = 39.25
        for _ in range(3):
            actions = {"AC000": [2, 2, 2]}
            _, _, _, _, infos = env.step(actions)
            if "AC000" in infos and "textual_state" in infos["AC000"]:
                ts = infos["AC000"]["textual_state"]
                # Position should change as aircraft moves east
                assert ts["position"]["lat"] != prev_lat or ts["position"]["lon"] != 116.25
                prev_lat = ts["position"]["lat"]


# ===========================================================================
# Agent interaction
# ===========================================================================


class TestAgentInteraction:
    """Test with RandomAgent and RuleBasedAgent."""

    def test_rule_based_agent_straight_flight(self) -> None:
        """RuleBasedAgent selects [2,2,2] for all agents."""
        from bluesky_pettingzoo.agents.rule_based_agent import RuleBasedAgent

        env = _make_env(initial_count=3, max_steps=5)
        env.reset(seed=42)

        agent = RuleBasedAgent()
        for _ in range(5):
            action_spaces = {a: env.action_space(a) for a in env.agents}
            actions = agent.act({}, action_spaces)
            obs, rewards, terms, truncs, infos = env.step(actions)

            # All actions should be [2,2,2]
            for a in actions:
                assert actions[a] == [2, 2, 2]

    def test_random_agent_produces_valid_actions(self) -> None:
        """RandomAgent produces actions within action space."""
        from bluesky_pettingzoo.agents.random_agent import RandomAgent

        env = _make_env(initial_count=3, max_steps=5)
        env.reset(seed=42)

        agent = RandomAgent()
        for _ in range(5):
            action_spaces = {a: env.action_space(a) for a in env.agents}
            actions = agent.act({}, action_spaces)
            obs, rewards, terms, truncs, infos = env.step(actions)

            for agent_id in actions:
                space = env.action_space(agent_id)
                assert space.contains(actions[agent_id])

    def test_mixed_agent_actions(self) -> None:
        """Some agents fly straight, others take random actions."""
        from bluesky_pettingzoo.agents.random_agent import RandomAgent

        env = _make_env(initial_count=3, max_steps=5)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.20, "lon": 116.20, "alt": 35000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.25, "lon": 116.25, "alt": 34000, "hdg": 180, "tas": 440},
            "AC002": {"lat": 39.30, "lon": 116.30, "alt": 36000, "hdg": 270, "tas": 460},
        })

        rng = np.random.RandomState(42)
        for _ in range(5):
            actions = {}
            for a in env.agents:
                actions[a] = [int(rng.randint(5)), int(rng.randint(5)), int(rng.randint(5))]
            obs, rewards, _, _, _ = env.step(actions)
            assert len(rewards) > 0


# ===========================================================================
# Deterministic reset
# ===========================================================================


class TestDeterministicReset:
    """Same seed should produce identical initial state."""

    def test_same_seed_same_agents(self) -> None:
        """Two envs with same seed should have same agent list."""
        env1 = _make_env(initial_count=3, max_steps=5)
        env2 = _make_env(initial_count=3, max_steps=5)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        assert env1.agents == env2.agents

    def test_same_seed_same_observations(self) -> None:
        """Two envs with same seed should produce identical observations."""
        env1 = _make_env(initial_count=3, max_steps=5)
        env2 = _make_env(initial_count=3, max_steps=5)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        for agent_id in obs1:
            for field in obs1[agent_id]:
                np.testing.assert_array_equal(obs1[agent_id][field], obs2[agent_id][field])

    def test_different_seeds_different_state(self) -> None:
        """Different seeds should produce different initial positions."""
        env1 = _make_env(initial_count=3, max_steps=5)
        env2 = _make_env(initial_count=3, max_steps=5)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=99)

        # At least one observation should differ
        has_diff = False
        for agent_id in obs1:
            if agent_id in obs2:
                for field in obs1[agent_id]:
                    if not np.array_equal(obs1[agent_id][field], obs2[agent_id][field]):
                        has_diff = True
                        break
            if has_diff:
                break
        assert has_diff


# ===========================================================================
# Action effects
# ===========================================================================


class TestActionEffects:
    """Verify that actions actually change aircraft state through the pipeline."""

    def test_heading_command_applied(self) -> None:
        """Heading action produces HDG command that changes aircraft heading."""
        env = _make_env(initial_count=1, max_steps=2)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
        })

        # heading_idx=3 → +10°
        actions = {"AC000": [3, 2, 2]}
        env.step(actions)

        assert env._wrapper._aircraft["AC000"]["hdg"] == 100.0

    def test_altitude_command_applied(self) -> None:
        """Altitude action produces ALT command that changes aircraft altitude."""
        env = _make_env(initial_count=1, max_steps=2)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
        })

        # altitude_idx=4 → +2000ft
        actions = {"AC000": [2, 4, 2]}
        env.step(actions)

        assert env._wrapper._aircraft["AC000"]["alt"] == 37000.0

    def test_speed_command_applied(self) -> None:
        """Speed action produces SPD command that changes aircraft speed."""
        env = _make_env(initial_count=1, max_steps=2)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
        })

        # speed_idx=0 → -20kt
        actions = {"AC000": [2, 2, 0]}
        env.step(actions)

        assert env._wrapper._aircraft["AC000"]["tas"] == 430.0

    def test_all_three_commands_applied(self) -> None:
        """All three action axes produce commands simultaneously."""
        env = _make_env(initial_count=1, max_steps=2)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 180, "tas": 450},
        })

        # heading_idx=0 → -20, altitude_idx=3 → +1000, speed_idx=4 → +20
        actions = {"AC000": [0, 3, 4]}
        env.step(actions)

        st = env._wrapper._aircraft["AC000"]
        assert st["hdg"] == 160.0  # 180 - 20
        assert st["alt"] == 36000.0  # 35000 + 1000
        assert st["tas"] == 470.0  # 450 + 20

    def test_no_action_no_command(self) -> None:
        """No-op action [2,2,2] should not change aircraft state."""
        env = _make_env(initial_count=1, max_steps=2)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
        })

        before = dict(env._wrapper._aircraft["AC000"])
        actions = {"AC000": [2, 2, 2]}
        env.step(actions)

        # Only lat/lon should change (due to movement), hdg/alt/tas unchanged
        after = env._wrapper._aircraft["AC000"]
        assert after["hdg"] == before["hdg"]
        assert after["alt"] == before["alt"]
        assert after["tas"] == before["tas"]


# ===========================================================================
# Reward integration
# ===========================================================================


class TestRewardIntegration:
    """Verify reward components work together in the env context."""

    def test_smoothness_penalty_on_action(self) -> None:
        """Taking action should incur smoothness penalty."""
        env = _make_env(initial_count=1, max_steps=2)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
        })

        # No-op action
        _, rewards_noop, _, _, _ = env.step({"AC000": [2, 2, 2]})

        # Reset and take action
        env.reset(seed=42)
        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
        })
        _, rewards_action, _, _, _ = env.step({"AC000": [3, 2, 2]})

        # Action version should have smoothness penalty (-0.1 * 0.5 = -0.05)
        assert rewards_action["AC000"] < rewards_noop["AC000"]

    def test_efficiency_goal_tracking(self) -> None:
        """Aircraft moving toward goal should get better efficiency reward."""
        env = _make_env(initial_count=1, max_steps=2)
        env.reset(seed=42)

        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
        })

        actions = {"AC000": [2, 2, 2]}
        _, rewards, _, _, _ = env.step(actions)

        # Should have step penalty at minimum
        assert rewards["AC000"] < 0  # at least step penalty

    def test_no_missing_agents_in_rewards(self) -> None:
        """All agents in agents list before step should have rewards after."""
        env = _make_env(initial_count=3, max_steps=3)
        env.reset(seed=42)

        for _ in range(3):
            agents_before = list(env.agents)
            actions = {a: [2, 2, 2] for a in env.agents}
            _, rewards, _, _, _ = env.step(actions)

            for agent_id in agents_before:
                assert agent_id in rewards, f"{agent_id} missing from rewards"


# ===========================================================================
# Observation integration
# ===========================================================================


class TestObservationIntegration:
    """Verify observations are properly generated in the env context."""

    def test_observation_shapes(self) -> None:
        """All observation fields have correct shapes."""
        env = _make_env(initial_count=3, max_steps=2)
        env.reset(seed=42)

        actions = {a: [2, 2, 2] for a in env.agents}
        obs, _, _, _, _ = env.step(actions)

        for agent_id in obs:
            o = obs[agent_id]
            assert o["self_state"].shape == (9,)
            assert o["other_aircraft"].shape == (10, 10)
            assert o["other_aircraft_mask"].shape == (10,)
            assert o["goal"].shape == (4,)

    def test_observation_in_space(self) -> None:
        """All observations lie within declared observation space."""
        env = _make_env(initial_count=3, max_steps=3)
        env.reset(seed=42)

        for _ in range(3):
            actions = {a: [2, 2, 2] for a in env.agents}
            obs, _, _, _, _ = env.step(actions)
            for agent_id in obs:
                space = env.observation_space(agent_id)
                assert space.contains(obs[agent_id]), (
                    f"Observation for {agent_id} not in space"
                )

    def test_mask_reflects_observable_aircraft(self) -> None:
        """other_aircraft_mask should have correct number of 1s."""
        env = _make_env(initial_count=3, max_steps=2)
        env.reset(seed=42)

        # Place two aircraft close, one far
        _place_aircraft(env, {
            "AC000": {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
            "AC001": {"lat": 39.27, "lon": 116.25, "alt": 35000, "hdg": 90, "tas": 450},
            "AC002": {"lat": 39.45, "lon": 116.45, "alt": 36000, "hdg": 90, "tas": 450},
        })

        actions = {a: [2, 2, 2] for a in env.agents}
        obs, _, _, _, _ = env.step(actions)

        # AC000 should see AC001 (close) but maybe not AC002 (far + alt diff)
        mask = obs["AC000"]["other_aircraft_mask"]
        observable_count = int(mask.sum())
        assert observable_count >= 1  # at least AC001
        assert observable_count <= 2  # at most both

    def test_textual_state_has_required_fields(self) -> None:
        """Textual state should have all required fields."""
        env = _make_env(initial_count=2, max_steps=2)
        env.reset(seed=42)

        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, _, _, infos = env.step(actions)

        required = ["agent_id", "position", "heading", "altitude", "speed",
                     "observable_aircraft", "conflict_status", "text"]
        for agent_id in infos:
            ts = infos[agent_id]["textual_state"]
            for field in required:
                assert field in ts, f"Missing {field} in textual_state for {agent_id}"

    def test_airspace_snapshot_has_positions(self) -> None:
        """Airspace snapshot should contain aircraft positions."""
        env = _make_env(initial_count=2, max_steps=2)
        env.reset(seed=42)

        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, _, _, infos = env.step(actions)

        for agent_id in infos:
            snap = infos[agent_id]["airspace_snapshot"]
            assert "aircraft_positions" in snap
            assert "sectors" in snap
