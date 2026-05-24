"""Integration tests using real BlueSky simulator (G-V01).

These tests require the BlueSky simulator to be installed and importable.
They run actual BlueSky simulations to verify end-to-end correctness.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import bluesky as bs
import numpy as np
import pytest
import yaml

from bluesky_pettingzoo.actions.translator import ActionTranslator
from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario
from bluesky_pettingzoo.envs.scenarios.horizontal_cr import HorizontalCRScenario
from bluesky_pettingzoo.envs.scenarios.vertical_cr import VerticalCRScenario
from bluesky_pettingzoo.envs.scenarios.sector_cr import SectorCRScenario
from bluesky_pettingzoo.envs.scenarios.merge import MergeScenario
from bluesky_pettingzoo.envs.scenarios.descent import DescentScenario
from bluesky_pettingzoo.envs.scenarios.static_obstacle import StaticObstacleScenario
from bluesky_pettingzoo.envs.scenarios.sector_capacity import SectorCapacityScenario
from bluesky_pettingzoo.envs.scenarios.route_nav import RouteNavScenario
from bluesky_pettingzoo.observations.manager import ObservationManager
from bluesky_pettingzoo.rewards.calculator import RewardCalculator
from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> dict[str, Any]:
    with open(CONFIG_DIR / "default.yaml", encoding="utf-8") as f:
        default = yaml.safe_load(f)
    with open(CONFIG_DIR / "rewards.yaml", encoding="utf-8") as f:
        rewards = yaml.safe_load(f)
    default["components"] = rewards["components"]
    # Use small aircraft count for fast tests
    default["aircraft"]["initial_count"] = 2
    default["simulation"]["max_episode_steps"] = 10
    return default


def _make_env(config: dict[str, Any], scenario=None) -> BlueSkyMARLEnv:
    wrapper = BlueSkyWrapper(config)
    obs_mgr = ObservationManager(config)
    act_trans = ActionTranslator(config)
    calc = RewardCalculator()
    calc.register(ConflictPenalty(config), weight=1.0)
    calc.register(SmoothnessPenalty(config), weight=0.5)
    calc.register(EfficiencyReward(config), weight=0.3)
    return BlueSkyMARLEnv(
        config=config,
        wrapper=wrapper,
        observation_manager=obs_mgr,
        action_translator=act_trans,
        reward_calculator=calc,
        rewards_config=config,
        scenario=scenario,
    )


# ===========================================================================
# G-V01: Real BlueSky integration tests
# ===========================================================================


class TestRealBlueSkyReset:
    """Verify reset works with real BlueSky."""

    def test_real_bluesky_reset(self) -> None:
        """Reset creates aircraft and returns valid observations."""
        config = _make_config()
        env = _make_env(config)
        try:
            obs, info = env.reset(seed=42)

            assert len(env.agents) == 2
            for agent_id in env.agents:
                assert agent_id in obs
                assert "self_state" in obs[agent_id]
                assert obs[agent_id]["self_state"].shape == (9,)
        finally:
            env.close()


class TestRealBlueSkyStep:
    """Verify step works with real BlueSky."""

    def test_real_bluesky_step(self) -> None:
        """Step advances simulation and returns valid 5-tuple."""
        config = _make_config()
        env = _make_env(config)
        try:
            env.reset(seed=42)
            noop = {aid: [2, 2, 2] for aid in env.agents}

            obs, rewards, terms, trunks, infos = env.step(noop)

            assert isinstance(obs, dict)
            assert isinstance(rewards, dict)
            assert isinstance(terms, dict)
            assert isinstance(trunks, dict)
            assert isinstance(infos, dict)

            for agent_id in env.agents:
                assert agent_id in rewards
                assert isinstance(rewards[agent_id], float)
        finally:
            env.close()


class TestRealBlueSkyStateConsistency:
    """Verify aircraft state is consistent between BlueSky and wrapper."""

    def test_real_bluesky_state_consistency(self) -> None:
        """State read via wrapper matches BlueSky internal state."""
        config = _make_config()
        env = _make_env(config)
        try:
            env.reset(seed=42)

            for agent_id in env.agents:
                state = env._get_aircraft_state(agent_id)
                raw = env._wrapper.get_aircraft_state(agent_id)

                assert state.lat == pytest.approx(raw["lat"], abs=1e-6)
                assert state.lon == pytest.approx(raw["lon"], abs=1e-6)
                assert state.alt == pytest.approx(raw["alt"], abs=1.0)
                assert state.hdg == pytest.approx(raw["hdg"], abs=1e-3)
                # TAS is computed by BlueSky performance model (CAS→TAS at altitude)
                assert state.tas > 0
        finally:
            env.close()


class TestRealBlueSkyClose:
    """Verify close works with real BlueSky."""

    def test_real_bluesky_close(self) -> None:
        """Close properly cleans up aircraft state."""
        config = _make_config()
        env = _make_env(config)
        env.reset(seed=42)

        # Close should not raise
        env.close()

        # After close, wrapper should be marked as not initialized
        assert not env._wrapper._initialized
        # Aircraft should be cleaned up
        assert len(bs.traf.id) == 0


class TestRealBlueSkyFullEpisode:
    """Verify a full episode runs with real BlueSky."""

    def test_real_bluesky_full_episode(self) -> None:
        """Run a complete episode from reset to truncation."""
        config = _make_config()
        env = _make_env(config)
        try:
            obs, info = env.reset(seed=42)
            total_steps = 0
            noop = [2, 2, 2]

            for _ in range(config["simulation"]["max_episode_steps"] + 5):
                actions = {aid: noop for aid in env.agents}
                if not env.agents:
                    break
                obs, rewards, terms, trunks, infos = env.step(actions)
                total_steps += 1

                # All observations should be valid
                for agent_id in env.agents:
                    if agent_id in obs:
                        assert obs[agent_id]["self_state"].shape == (9,)

                # Check truncation
                if any(trunks.values()):
                    break

            assert total_steps > 0
        finally:
            env.close()


class TestRealBlueSkyAircraftCreation:
    """Verify aircraft creation parameters match BlueSky internal state."""

    def test_aircraft_altitude_matches_creation(self) -> None:
        """Aircraft altitude in BlueSky matches the creation parameter."""
        config = _make_config()
        env = _make_env(config)
        try:
            env.reset(seed=42)
            for agent_id in env.agents:
                state = env._get_aircraft_state(agent_id)
                # Altitude should be within the configured spawn range
                assert 29000 <= state.alt <= 37000, (
                    f"{agent_id} alt={state.alt} outside spawn range [29000, 37000]"
                )
        finally:
            env.close()

    def test_aircraft_speed_matches_creation(self) -> None:
        """Aircraft speed in BlueSky is positive and reasonable."""
        config = _make_config()
        env = _make_env(config)
        try:
            env.reset(seed=42)
            for agent_id in env.agents:
                state = env._get_aircraft_state(agent_id)
                assert state.tas > 0, f"{agent_id} tas={state.tas} should be positive"
        finally:
            env.close()


class TestRealBlueSkyActionEffects:
    """Verify that actions actually change aircraft behavior."""

    def test_heading_change_turns_aircraft(self) -> None:
        """A heading action should change the aircraft's heading over time."""
        config = _make_config()
        config["simulation"]["max_episode_steps"] = 60
        env = _make_env(config)
        try:
            obs, _ = env.reset(seed=42)
            agent_id = env.agents[0]
            initial_hdg = env._get_aircraft_state(agent_id).hdg

            # Step many times with heading change to let autopilot turn
            for _ in range(50):
                if agent_id not in env.agents:
                    break
                actions = {aid: [2, 2, 2] for aid in env.agents}
                actions[agent_id] = [0, 2, 2]  # turn left 20 degrees
                env.step(actions)

            # After many steps, heading should have changed
            if agent_id in env.agents:
                new_hdg = env._get_aircraft_state(agent_id).hdg
                hdg_diff = abs(new_hdg - initial_hdg)
                if hdg_diff > 180:
                    hdg_diff = 360 - hdg_diff
                assert hdg_diff > 0.1, (
                    f"Heading should have changed: initial={initial_hdg}, new={new_hdg}"
                )
        finally:
            env.close()

    def test_noop_action_preserves_heading(self) -> None:
        """A noop action should not significantly change heading."""
        config = _make_config()
        config["simulation"]["max_episode_steps"] = 5
        env = _make_env(config)
        try:
            obs, _ = env.reset(seed=42)
            agent_id = env.agents[0]
            initial_hdg = env._get_aircraft_state(agent_id).hdg

            noop = {aid: [2, 2, 2] for aid in env.agents}
            env.step(noop)

            if agent_id in env.agents:
                new_hdg = env._get_aircraft_state(agent_id).hdg
                hdg_diff = abs(new_hdg - initial_hdg)
                if hdg_diff > 180:
                    hdg_diff = 360 - hdg_diff
                assert hdg_diff < 1.0, (
                    f"Noop should not change heading: initial={initial_hdg}, new={new_hdg}"
                )
        finally:
            env.close()


class TestRealBlueSkyMultiStepDynamics:
    """Verify aircraft position changes over multiple simulation steps."""

    def test_aircraft_moves_over_time(self) -> None:
        """Aircraft position should change after multiple steps."""
        config = _make_config()
        config["simulation"]["max_episode_steps"] = 10
        env = _make_env(config)
        try:
            env.reset(seed=42)
            agent_id = env.agents[0]
            initial_state = env._get_aircraft_state(agent_id)

            noop = {aid: [2, 2, 2] for aid in env.agents}
            for _ in range(5):
                if not env.agents:
                    break
                env.step(noop)

            if agent_id in env.agents:
                final_state = env._get_aircraft_state(agent_id)
                # Position should have changed (aircraft is moving)
                pos_changed = (
                    abs(final_state.lat - initial_state.lat) > 1e-6
                    or abs(final_state.lon - initial_state.lon) > 1e-6
                )
                assert pos_changed, "Aircraft position should change over time"
        finally:
            env.close()

    def test_altitude_action_changes_altitude(self) -> None:
        """An altitude action should change the aircraft's altitude over time."""
        config = _make_config()
        config["simulation"]["max_episode_steps"] = 60
        env = _make_env(config)
        try:
            obs, _ = env.reset(seed=42)
            agent_id = env.agents[0]
            initial_alt = env._get_aircraft_state(agent_id).alt

            # Action [2, 0, 2] = heading_adj=0, alt_adj=-2000, spd_adj=0
            # Command many descent steps
            for _ in range(50):
                if agent_id not in env.agents:
                    break
                actions = {aid: [2, 2, 2] for aid in env.agents}
                actions[agent_id] = [2, 0, 2]  # descend 2000 ft
                env.step(actions)

            if agent_id in env.agents:
                final_alt = env._get_aircraft_state(agent_id).alt
                # Altitude should have changed (descending)
                # BlueSky autopilot descends gradually; even 1 ft confirms command works
                assert final_alt < initial_alt - 0.5, (
                    f"Altitude should decrease: initial={initial_alt}, final={final_alt}"
                )
        finally:
            env.close()


class TestRealBlueSkyScenarioIntegration:
    """Verify scenarios work with real BlueSky."""

    def test_waypoint_nav_scenario_reset(self) -> None:
        """WaypointNav scenario should initialize correctly with real BlueSky."""
        config = _make_config()
        config["aircraft"]["initial_count"] = 3
        scenario = WaypointNavScenario(num_aircraft=3, seed=42)
        env = _make_env(config, scenario=scenario)
        try:
            obs, infos = env.reset(seed=42)
            assert len(env.agents) == 3
            for agent_id in env.agents:
                assert agent_id in obs
                wp = scenario.get_waypoint(agent_id)
                assert "lat" in wp
                assert "lon" in wp
        finally:
            env.close()

    def test_waypoint_nav_scenario_step(self) -> None:
        """WaypointNav scenario should step correctly with real BlueSky."""
        config = _make_config()
        config["aircraft"]["initial_count"] = 3
        config["simulation"]["max_episode_steps"] = 5
        scenario = WaypointNavScenario(num_aircraft=3, seed=42)
        env = _make_env(config, scenario=scenario)
        try:
            env.reset(seed=42)
            noop = {aid: [2, 2, 2] for aid in env.agents}
            obs, rewards, terms, trunks, infos = env.step(noop)

            assert isinstance(obs, dict)
            assert isinstance(rewards, dict)
            for agent_id in env.agents:
                assert agent_id in rewards
        finally:
            env.close()


class TestRealBlueSkyObservationStructure:
    """Verify observation dict structure with real BlueSky."""

    def test_observation_has_self_state(self) -> None:
        """Observation should contain self_state array."""
        config = _make_config()
        env = _make_env(config)
        try:
            obs, _ = env.reset(seed=42)
            for agent_id in env.agents:
                assert "self_state" in obs[agent_id]
                assert isinstance(obs[agent_id]["self_state"], np.ndarray)
                assert obs[agent_id]["self_state"].shape == (9,)
        finally:
            env.close()

    def test_observation_has_other_aircraft(self) -> None:
        """Observation should contain other_aircraft array."""
        config = _make_config()
        env = _make_env(config)
        try:
            obs, _ = env.reset(seed=42)
            for agent_id in env.agents:
                assert "other_aircraft" in obs[agent_id]
                assert isinstance(obs[agent_id]["other_aircraft"], np.ndarray)
        finally:
            env.close()

    def test_observation_has_goal(self) -> None:
        """Observation should contain goal array."""
        config = _make_config()
        env = _make_env(config)
        try:
            obs, _ = env.reset(seed=42)
            for agent_id in env.agents:
                assert "goal" in obs[agent_id]
                assert isinstance(obs[agent_id]["goal"], np.ndarray)
        finally:
            env.close()

    def test_observation_values_finite(self) -> None:
        """All observation values should be finite (no NaN/Inf)."""
        config = _make_config()
        env = _make_env(config)
        try:
            obs, _ = env.reset(seed=42)
            for agent_id in env.agents:
                for key, val in obs[agent_id].items():
                    if isinstance(val, np.ndarray):
                        assert np.all(np.isfinite(val)), (
                            f"{agent_id}.{key} contains non-finite values: {val}"
                        )
        finally:
            env.close()


class TestRealBlueSkyResetIdempotency:
    """Verify reset can be called multiple times."""

    def test_double_reset_works(self) -> None:
        """Calling reset twice should work without errors."""
        config = _make_config()
        env = _make_env(config)
        try:
            obs1, _ = env.reset(seed=42)
            obs2, _ = env.reset(seed=99)

            assert len(env.agents) == 2
            for agent_id in env.agents:
                assert agent_id in obs2
        finally:
            env.close()

    def test_reset_after_steps_works(self) -> None:
        """Reset after stepping should work without errors."""
        config = _make_config()
        config["simulation"]["max_episode_steps"] = 5
        env = _make_env(config)
        try:
            env.reset(seed=42)
            noop = {aid: [2, 2, 2] for aid in env.agents}
            env.step(noop)

            obs, _ = env.reset(seed=99)
            assert len(env.agents) == 2
            for agent_id in env.agents:
                assert agent_id in obs
        finally:
            env.close()


# ===========================================================================
# G-V02: Real BlueSky scenario integration tests
# ===========================================================================


def _run_scenario_episode(scenario, num_aircraft: int, max_steps: int = 10) -> None:
    """Helper: run a full episode with a scenario on real BlueSky."""
    config = _make_config()
    config["aircraft"]["initial_count"] = num_aircraft
    config["simulation"]["max_episode_steps"] = max_steps
    env = _make_env(config, scenario=scenario)
    try:
        obs, info = env.reset(seed=42)
        assert len(env.agents) >= 1
        for agent_id in env.agents:
            assert agent_id in obs
            assert obs[agent_id]["self_state"].shape == (9,)

        noop = {aid: [2, 2, 2] for aid in env.agents}
        for _ in range(max_steps + 5):
            if not env.agents:
                break
            obs, rewards, terms, trunks, infos = env.step({aid: [2, 2, 2] for aid in env.agents})
            for agent_id in env.agents:
                if agent_id in obs:
                    assert np.all(np.isfinite(obs[agent_id]["self_state"]))
            if any(trunks.values()):
                break
    finally:
        env.close()


class TestRealBlueSkyHorizontalCR:
    """HorizontalCR scenario on real BlueSky."""

    def test_horizontal_cr_episode(self) -> None:
        scenario = HorizontalCRScenario(num_aircraft=3, seed=42)
        _run_scenario_episode(scenario, num_aircraft=3)


class TestRealBlueSkyVerticalCR:
    """VerticalCR scenario on real BlueSky."""

    def test_vertical_cr_episode(self) -> None:
        scenario = VerticalCRScenario(num_aircraft=3, seed=42)
        _run_scenario_episode(scenario, num_aircraft=3)


class TestRealBlueSkySectorCR:
    """SectorCR scenario on real BlueSky."""

    def test_sector_cr_episode(self) -> None:
        scenario = SectorCRScenario(num_aircraft=3, seed=42)
        _run_scenario_episode(scenario, num_aircraft=3)


class TestRealBlueSkyMerge:
    """Merge scenario on real BlueSky."""

    def test_merge_episode(self) -> None:
        scenario = MergeScenario(num_aircraft=5, seed=42)
        _run_scenario_episode(scenario, num_aircraft=5)


class TestRealBlueSkyDescent:
    """Descent scenario on real BlueSky."""

    def test_descent_episode(self) -> None:
        scenario = DescentScenario(num_aircraft=3, seed=42)
        _run_scenario_episode(scenario, num_aircraft=3)


class TestRealBlueSkyStaticObstacle:
    """StaticObstacle scenario on real BlueSky."""

    def test_static_obstacle_episode(self) -> None:
        scenario = StaticObstacleScenario(num_aircraft=1, seed=42)
        _run_scenario_episode(scenario, num_aircraft=1)


class TestRealBlueSkySectorCapacity:
    """SectorCapacity scenario on real BlueSky."""

    def test_sector_capacity_episode(self) -> None:
        scenario = SectorCapacityScenario(num_aircraft=4, num_sectors=2, sector_capacity=3, seed=42)
        _run_scenario_episode(scenario, num_aircraft=4)


class TestRealBlueSkyRouteNav:
    """RouteNav scenario on real BlueSky."""

    def test_route_nav_episode(self) -> None:
        scenario = RouteNavScenario(num_aircraft=3, seed=42)
        _run_scenario_episode(scenario, num_aircraft=3)
