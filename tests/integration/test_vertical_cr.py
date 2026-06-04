"""Tests for VerticalCRScenario (T-V10).

Vertical conflict resolution: multiple aircraft at similar horizontal positions
but different altitudes, use vertical speed maneuvers to avoid conflicts.
Conflict requires BOTH horizontal < 5 NM AND vertical < 1000 ft.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.vertical_cr import VerticalCRScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty
from bluesky_pettingzoo.utils.geometry import haversine_distance
from tests.helpers.env_factory import make_config as _make_config
from tests.helpers.env_factory import write_rewards_yaml as _write_rewards_yaml

# ---------------------------------------------------------------------------
# Fake BlueSkyWrapper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env_with_scenario(
    env_config: dict[str, Any],
    scenario: VerticalCRScenario,
) -> BlueSkyMARLEnv:
    """Create a BlueSkyMARLEnv with a VerticalCRScenario."""
    wrapper = BlueSkyWrapper(env_config)
    obs_manager = ObservationManager(env_config)
    action_translator = ActionTranslator(env_config)

    rewards_path = env_config["_rewards_yaml"]
    with open(rewards_path, encoding="utf-8") as f:
        rewards_cfg = yaml.safe_load(f)
    merged = {**env_config, **rewards_cfg}

    calc = RewardCalculator()
    calc.register(ConflictPenalty(merged), weight=1.0)
    calc.register(SmoothnessPenalty(merged), weight=0.5)
    eff = EfficiencyReward(merged)
    calc.register(eff, weight=0.3)

    return BlueSkyMARLEnv(
        config=env_config,
        wrapper=wrapper,
        observation_manager=obs_manager,
        action_translator=action_translator,
        reward_calculator=calc,
        rewards_config=rewards_cfg,
        scenario=scenario,
    )


# ===========================================================================
# T-V10 tests
# ===========================================================================


class TestVerticalCRSetup:
    """Scenario initialization should succeed."""

    def test_vertical_cr_setup(self, tmp_path: Path) -> None:
        """setup() returns the configured number of agent IDs."""
        scenario = VerticalCRScenario(num_aircraft=3, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        agents = scenario.setup(rng, bounds)

        assert len(agents) == 3
        assert all(isinstance(a, str) for a in agents)
        assert len(set(agents)) == 3

        # All aircraft should be at different altitudes
        alts = [scenario.get_waypoint(a)["alt"] for a in agents]
        assert len(set(alts)) == len(alts), "Aircraft should be at different altitudes"


class TestVerticalCRConflictBoth:
    """Horizontal + vertical both violated = conflict."""

    def test_vertical_cr_conflict_both(self, tmp_path: Path) -> None:
        """Two aircraft close horizontally AND vertically detected as conflict."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=2)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = VerticalCRScenario(num_aircraft=2, seed=42)
        env = _make_env_with_scenario(config, scenario)
        env.reset(seed=42)

        agents = list(env.agents)
        wrapper = env._wrapper

        # Position aircraft close horizontally (3 NM) and vertically (500 ft)
        wrapper.set_aircraft_state(agents[0], lat=39.25, lon=116.25, alt=35000, hdg=90.0)
        wrapper.set_aircraft_state(agents[1], lat=39.25, lon=116.30, alt=34500, hdg=270.0)

        h_dist = haversine_distance(39.25, 116.25, 39.25, 116.30)
        v_dist = abs(35000 - 34500)
        assert h_dist < 5.0  # Within NMAC horizontal
        assert v_dist < 1000  # Within NMAC vertical

        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, _, _, infos = env.step(actions)

        for agent_id in agents:
            if agent_id in infos:
                ts = infos[agent_id].get("textual_state", {})
                assert ts.get("conflict_status") in ("warning", "nmac")


class TestVerticalCRHorizontalOnly:
    """Only horizontal violation (vertical separated) ≠ conflict."""

    def test_vertical_cr_horizontal_only(self, tmp_path: Path) -> None:
        """Aircraft close horizontally but well separated vertically → safe."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=2)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = VerticalCRScenario(num_aircraft=2, seed=42)
        env = _make_env_with_scenario(config, scenario)
        env.reset(seed=42)

        agents = list(env.agents)
        wrapper = env._wrapper

        # Close horizontally (3 NM) but far vertically (5000 ft)
        wrapper.set_aircraft_state(agents[0], lat=39.25, lon=116.25, alt=35000, hdg=90.0)
        wrapper.set_aircraft_state(agents[1], lat=39.25, lon=116.30, alt=30000, hdg=270.0)

        h_dist = haversine_distance(39.25, 116.25, 39.25, 116.30)
        v_dist = abs(35000 - 30000)
        assert h_dist < 5.0  # Within horizontal threshold
        assert v_dist > 1000  # Beyond vertical threshold

        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, _, _, infos = env.step(actions)

        for agent_id in agents:
            if agent_id in infos:
                ts = infos[agent_id].get("textual_state", {})
                assert ts.get("conflict_status") == "safe"


class TestVerticalCRVerticalOnly:
    """Only vertical violation (horizontally separated) ≠ conflict."""

    def test_vertical_cr_vertical_only(self, tmp_path: Path) -> None:
        """Aircraft close vertically but well separated horizontally → safe."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=2)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = VerticalCRScenario(num_aircraft=2, seed=42)
        env = _make_env_with_scenario(config, scenario)
        env.reset(seed=42)

        agents = list(env.agents)
        wrapper = env._wrapper

        # Close vertically (500 ft) but far horizontally (20 NM)
        wrapper.set_aircraft_state(agents[0], lat=39.1, lon=116.1, alt=35000, hdg=90.0)
        wrapper.set_aircraft_state(agents[1], lat=39.4, lon=116.4, alt=34500, hdg=270.0)

        h_dist = haversine_distance(39.1, 116.1, 39.4, 116.4)
        v_dist = abs(35000 - 34500)
        assert h_dist > 5.0  # Beyond horizontal threshold
        assert v_dist < 1000  # Within vertical threshold

        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, _, _, infos = env.step(actions)

        for agent_id in agents:
            if agent_id in infos:
                ts = infos[agent_id].get("textual_state", {})
                assert ts.get("conflict_status") == "safe"


class TestVerticalCRActionVSOnly:
    """VerticalCR scenario should restrict action space to vertical speed only."""

    def test_vertical_cr_action_vs_only(self) -> None:
        """Scenario reports vertical-speed-only action constraints."""
        scenario = VerticalCRScenario(num_aircraft=3, seed=42)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 39.5, "lon_min": 116.0, "lon_max": 116.5}
        scenario.setup(rng, bounds)

        # action_dimensions[1] = altitude/vertical speed index only
        assert scenario.action_dimensions == [1]


class TestVerticalCRFullEpisode:
    """Full episode with VerticalCRScenario should run without errors."""

    def test_vertical_cr_full_episode(self, tmp_path: Path) -> None:
        """Run a complete episode with the scenario."""
        _write_rewards_yaml(tmp_path)
        config = _make_config(initial_count=3, max_steps=20)
        config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")

        scenario = VerticalCRScenario(num_aircraft=3, seed=42)
        env = _make_env_with_scenario(config, scenario)
        obs, infos = env.reset(seed=42)

        assert len(env.agents) == 3

        total_reward = 0.0
        for step in range(20):
            actions = {a: [2, 2, 2] for a in env.agents}
            if not actions:
                break
            obs, rewards, terminations, truncations, infos = env.step(actions)
            total_reward += sum(rewards.values())

            if not env.agents:
                break

        assert np.isfinite(total_reward)
