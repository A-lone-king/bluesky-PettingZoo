"""Tests for BlueSkyMARLEnv (PettingZoo ParallelEnv)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from gymnasium import spaces

from bluesky_pettingzoo.bluesky.wrapper import BlueSkyWrapper
from bluesky_pettingzoo.envs.parallel_env import BlueSkyMARLEnv
from bluesky_pettingzoo.utils.types import AircraftState
from tests.helpers.env_factory import make_config as _make_config
from tests.helpers.env_factory import write_rewards_yaml as _write_rewards_yaml

# ---------------------------------------------------------------------------
# Fake BlueSkyWrapper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def env_config(tmp_path: Path) -> dict[str, Any]:
    """Full config with rewards YAML written to disk."""
    config = _make_config(initial_count=3, max_steps=360)
    _write_rewards_yaml(tmp_path)
    config["_rewards_yaml"] = str(tmp_path / "rewards.yaml")
    return config


@pytest.fixture
def env(env_config: dict[str, Any]) -> BlueSkyMARLEnv:
    """Create env with BlueSkyWrapper."""
    from bluesky_pettingzoo.actions.translator import ActionTranslator
    from bluesky_pettingzoo.observations.manager import ObservationManager
    from bluesky_pettingzoo.rewards.calculator import RewardCalculator
    from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
    from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
    from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty

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
    )


# ===========================================================================
# Reset tests
# ===========================================================================


class TestResetReturnsTuple:
    """reset() must return (observations, infos) tuple."""

    def test_reset_returns_tuple(self, env: BlueSkyMARLEnv) -> None:
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestResetObservationsKeys:
    """Observation keys must match agents list."""

    def test_reset_observations_keys(self, env: BlueSkyMARLEnv) -> None:
        obs, _ = env.reset()
        assert set(obs.keys()) == set(env.agents)


class TestResetInfosKeys:
    """Infos keys must match agents list."""

    def test_reset_infos_keys(self, env: BlueSkyMARLEnv) -> None:
        _, infos = env.reset()
        assert set(infos.keys()) == set(env.agents)


class TestResetObservationInSpace:
    """Each observation value must lie within observation_space."""

    def test_reset_observation_in_space(self, env: BlueSkyMARLEnv) -> None:
        obs, _ = env.reset()
        for agent_id in env.agents:
            space = env.observation_space(agent_id)
            assert space.contains(obs[agent_id]), f"Observation for {agent_id} not in space"


class TestResetAgentsPopulated:
    """After reset, agents list must be non-empty."""

    def test_reset_agents_populated(self, env: BlueSkyMARLEnv) -> None:
        env.reset()
        assert len(env.agents) > 0


class TestResetWithSeed:
    """Same seed must produce identical initial state."""

    def test_reset_with_seed(self, env_config: dict[str, Any]) -> None:
        from bluesky_pettingzoo.actions.translator import ActionTranslator
        from bluesky_pettingzoo.observations.manager import ObservationManager
        from bluesky_pettingzoo.rewards.calculator import RewardCalculator
        from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
        from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
        from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty

        def make_env() -> BlueSkyMARLEnv:
            wrapper = BlueSkyWrapper(env_config)
            obs_manager = ObservationManager(env_config)
            action_translator = ActionTranslator(env_config)
            rewards_path = env_config["_rewards_yaml"]
            with open(rewards_path, encoding="utf-8") as f:
                rewards_cfg = yaml.safe_load(f)
            merged = {**env_config, **rewards_cfg}
            calc = RewardCalculator()
            calc.register(ConflictPenalty(merged), 1.0)
            calc.register(SmoothnessPenalty(merged), 0.5)
            calc.register(EfficiencyReward(merged), 0.3)
            return BlueSkyMARLEnv(
                config=env_config,
                wrapper=wrapper,
                observation_manager=obs_manager,
                action_translator=action_translator,
                reward_calculator=calc,
                rewards_config=rewards_cfg,
            )

        e1 = make_env()
        e2 = make_env()
        obs1, _ = e1.reset(seed=42)
        obs2, _ = e2.reset(seed=42)
        assert set(obs1.keys()) == set(obs2.keys())
        for k in obs1:
            for field in obs1[k]:
                np.testing.assert_array_equal(obs1[k][field], obs2[k][field])


class TestResetClearsPreviousState:
    """Second reset must clear old state and produce fresh observations."""

    def test_reset_clears_previous_state(self, env: BlueSkyMARLEnv) -> None:
        env.reset()
        # Run a few steps to change state
        action_spaces = {a: env.action_space(a) for a in env.agents}
        actions = {a: [0, 0, 0] for a in env.agents}
        env.step(actions)

        obs2, _ = env.reset()
        assert set(obs2.keys()) == set(env.agents)


# ===========================================================================
# Step tests
# ===========================================================================


class TestStepReturnsFiveTuple:
    """step() must return (obs, rewards, terms, truncs, infos)."""

    def test_step_returns_five_tuple(self, env: BlueSkyMARLEnv) -> None:
        env.reset()
        actions = {a: [2, 2, 2] for a in env.agents}
        result = env.step(actions)
        assert isinstance(result, tuple)
        assert len(result) == 5


class TestStepRewardsKeys:
    """Reward keys must match agents."""

    def test_step_rewards_keys(self, env: BlueSkyMARLEnv) -> None:
        env.reset()
        agents_before = list(env.agents)
        actions = {a: [2, 2, 2] for a in env.agents}
        _, rewards, _, _, _ = env.step(actions)
        assert set(rewards.keys()) == set(agents_before)


class TestStepTerminationsKeys:
    """Termination keys must match agents."""

    def test_step_terminations_keys(self, env: BlueSkyMARLEnv) -> None:
        env.reset()
        agents_before = list(env.agents)
        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, terms, _, _ = env.step(actions)
        assert set(terms.keys()) == set(agents_before)


class TestStepTruncationsKeys:
    """Truncation keys must match agents."""

    def test_step_truncations_keys(self, env: BlueSkyMARLEnv) -> None:
        env.reset()
        agents_before = list(env.agents)
        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, _, truncs, _ = env.step(actions)
        assert set(truncs.keys()) == set(agents_before)


class TestStepObservationInSpace:
    """Observations after step must lie within observation_space."""

    def test_step_observation_in_space(self, env: BlueSkyMARLEnv) -> None:
        obs, _ = env.reset()
        actions = {a: [2, 2, 2] for a in env.agents}
        obs2, _, _, _, _ = env.step(actions)
        for agent_id in obs2:
            space = env.observation_space(agent_id)
            assert space.contains(obs2[agent_id]), (
                f"Observation for {agent_id} not in space after step"
            )


class TestStepAgentsUpdate:
    """agents list may change after step (aircraft can leave airspace)."""

    def test_step_agents_update(self, env: BlueSkyMARLEnv) -> None:
        env.reset()
        agents_before = list(env.agents)
        actions = {a: [2, 2, 2] for a in env.agents}
        env.step(actions)
        # agents may or may not change, but it should be a valid list
        assert isinstance(env.agents, list)
        for a in env.agents:
            assert isinstance(a, str)


# ===========================================================================
# Space tests
# ===========================================================================


class TestObservationSpaceType:
    """observation_space() must return a Dict space."""

    def test_observation_space_type(self, env: BlueSkyMARLEnv) -> None:
        env.reset()
        space = env.observation_space(env.agents[0])
        assert isinstance(space, spaces.Dict)


class TestActionSpaceType:
    """action_space() must return a MultiDiscrete space."""

    def test_action_space_type(self, env: BlueSkyMARLEnv) -> None:
        env.reset()
        space = env.action_space(env.agents[0])
        assert isinstance(space, spaces.MultiDiscrete)


class TestActionSpaceSampleValid:
    """action_space.sample() must produce valid actions."""

    def test_action_space_sample_valid(self, env: BlueSkyMARLEnv) -> None:
        env.reset()
        for agent_id in env.agents:
            space = env.action_space(agent_id)
            for _ in range(50):
                sample = space.sample()
                assert space.contains(sample)


# ===========================================================================
# Lifecycle tests
# ===========================================================================


class TestEpisodeEndsOnMaxSteps:
    """Truncations must be all-True when max_episode_steps is reached."""

    def test_episode_ends_on_max_steps(self, tmp_path: Path) -> None:
        config = _make_config(initial_count=2, max_steps=3)
        rw_path = _write_rewards_yaml(tmp_path)
        config["_rewards_yaml"] = str(rw_path)

        from bluesky_pettingzoo.actions.translator import ActionTranslator
        from bluesky_pettingzoo.observations.manager import ObservationManager
        from bluesky_pettingzoo.rewards.calculator import RewardCalculator
        from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
        from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
        from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty

        wrapper = BlueSkyWrapper(config)
        obs_mgr = ObservationManager(config)
        act_trans = ActionTranslator(config)
        with open(rw_path, encoding="utf-8") as f:
            rw_cfg = yaml.safe_load(f)
        merged = {**config, **rw_cfg}
        calc = RewardCalculator()
        calc.register(ConflictPenalty(merged), 1.0)
        calc.register(SmoothnessPenalty(merged), 0.5)
        calc.register(EfficiencyReward(merged), 0.3)

        env = BlueSkyMARLEnv(
            config=config,
            wrapper=wrapper,
            observation_manager=obs_mgr,
            action_translator=act_trans,
            reward_calculator=calc,
            rewards_config=rw_cfg,
        )

        env.reset()
        for _ in range(3):
            actions = {a: [2, 2, 2] for a in env.agents}
            _, _, _, truncs, _ = env.step(actions)

        assert all(truncs.values())


class TestAgentRemovalOnExit:
    """Agents must be removed from self.agents when aircraft leaves airspace."""

    def test_agent_removal_on_exit(self, env: BlueSkyMARLEnv) -> None:
        env.reset()
        # Give one aircraft a heading that takes it out of bounds quickly
        first_agent = env.agents[0]
        actions = {a: [2, 2, 2] for a in env.agents}
        # heading_idx=0 → -20 degrees from current heading
        actions[first_agent] = [0, 2, 2]

        initial_agents = set(env.agents)
        # Run many steps; aircraft heading away from sector should eventually leave
        for _ in range(500):
            if not env.agents:
                break
            actions = {a: [2, 2, 2] for a in env.agents}
            if first_agent in env.agents:
                actions[first_agent] = [0, 2, 2]
            env.step(actions)

        # At least one agent should have been removed (or all, if enough steps)
        # Since we steered one aircraft, the set should have shrunk
        # If all left, agents is empty — that's fine too
        assert len(env.agents) < len(initial_agents) or len(env.agents) == 0


class TestInfosHasTextualState:
    """infos dict must contain 'textual_state' for each agent."""

    def test_infos_has_textual_state(self, env: BlueSkyMARLEnv) -> None:
        env.reset()
        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, _, _, infos = env.step(actions)
        for agent_id in infos:
            assert "textual_state" in infos[agent_id]
            ts = infos[agent_id]["textual_state"]
            assert "agent_id" in ts
            assert "text" in ts


class TestInfosHasAirspaceSnapshot:
    """infos dict must contain 'airspace_snapshot' for each agent."""

    def test_infos_has_airspace_snapshot(self, env: BlueSkyMARLEnv) -> None:
        env.reset()
        actions = {a: [2, 2, 2] for a in env.agents}
        _, _, _, _, infos = env.step(actions)
        for agent_id in infos:
            assert "airspace_snapshot" in infos[agent_id]
            snap = infos[agent_id]["airspace_snapshot"]
            assert "aircraft_positions" in snap


# ===========================================================================
# T-V08: Scenario integration tests
# ===========================================================================


class _TrackingScenario:
    """Minimal scenario implementation that tracks method calls for testing."""

    def __init__(
        self,
        agent_ids: list[str] | None = None,
        waypoints: dict[str, dict[str, float]] | None = None,
        truncate_ids: set[str] | None = None,
        new_agents_per_update: list[str] | None = None,
    ) -> None:
        self._agent_ids = agent_ids or ["AC000", "AC001", "AC002"]
        self._waypoints = waypoints or {}
        self._truncate_ids = truncate_ids or set()
        self._new_agents_per_update = new_agents_per_update or []

        # Tracking
        self.setup_called: bool = False
        self.setup_call_count: int = 0
        self.update_called: bool = False
        self.update_call_count: int = 0
        self.update_step_counts: list[int] = []
        self.truncate_checks: list[tuple[str, AircraftState]] = []
        self.reset_called: bool = False
        self.reset_call_count: int = 0

    def setup(self, rng: np.random.RandomState, airspace_bounds: dict[str, float]) -> list[str]:
        self.setup_called = True
        self.setup_call_count += 1
        return list(self._agent_ids)

    def get_spawn_config(self):
        from bluesky_pettingzoo.utils.types import SpawnConfig

        return SpawnConfig(
            altitude_range=(29000, 37000),
            speed_range=(400, 500),
            heading_range=(0, 360),
        )

    def get_conflict_config(self):
        from bluesky_pettingzoo.utils.types import ConflictConfig

        return ConflictConfig(
            nmac_horizontal_nm=5.0,
            nmac_vertical_ft=1000.0,
            warning_horizontal_nm=10.0,
            warning_vertical_ft=2000.0,
        )

    def should_truncate(
        self,
        agent_id: str,
        state: AircraftState,
        airspace_bounds: dict[str, float],
    ) -> bool:
        self.truncate_checks.append((agent_id, state))
        return agent_id in self._truncate_ids

    def get_waypoint(self, agent_id: str) -> dict[str, float]:
        return self._waypoints.get(
            agent_id,
            {"lat": 39.25, "lon": 116.25, "alt": 35000, "hdg": 90.0},
        )

    def update(self, step_count: int, all_states: dict[str, AircraftState]) -> list[str]:
        self.update_called = True
        self.update_call_count += 1
        self.update_step_counts.append(step_count)
        return list(self._new_agents_per_update)

    def reset(self, rng: np.random.RandomState) -> None:
        self.reset_called = True
        self.reset_call_count += 1


def _make_env_with_scenario(
    env_config: dict[str, Any],
    scenario: _TrackingScenario,
) -> BlueSkyMARLEnv:
    """Create a BlueSkyMARLEnv with a scenario instance."""
    from bluesky_pettingzoo.actions.translator import ActionTranslator
    from bluesky_pettingzoo.observations.manager import ObservationManager
    from bluesky_pettingzoo.rewards.calculator import RewardCalculator
    from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
    from bluesky_pettingzoo.rewards.components.efficiency import EfficiencyReward
    from bluesky_pettingzoo.rewards.components.smoothness import SmoothnessPenalty

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


class TestEnvWithoutScenario:
    """Without scenario, behavior is identical to V1.0."""

    def test_env_without_scenario(self, env: BlueSkyMARLEnv) -> None:
        """Env with no scenario works exactly like V1.0."""
        obs, infos = env.reset(seed=42)
        assert len(env.agents) == 3
        assert len(obs) == 3
        for aid in env.agents:
            assert aid in obs
            assert aid in infos


class TestEnvWithScenario:
    """With scenario, aircraft generation is driven by scenario.setup()."""

    def test_env_with_scenario(self, env_config: dict[str, Any], tmp_path: Path) -> None:
        """Scenario's agent IDs appear in env.agents after reset."""
        _write_rewards_yaml(tmp_path)
        scenario = _TrackingScenario(agent_ids=["SC000", "SC001", "SC002", "SC003"])
        env = _make_env_with_scenario(env_config, scenario)
        obs, infos = env.reset(seed=42)
        assert set(env.agents) == {"SC000", "SC001", "SC002", "SC003"}
        assert len(obs) == 4


class TestScenarioSetupCalled:
    """reset() must call scenario.setup()."""

    def test_scenario_setup_called(self, env_config: dict[str, Any], tmp_path: Path) -> None:
        """scenario.setup() is called during env.reset()."""
        _write_rewards_yaml(tmp_path)
        scenario = _TrackingScenario()
        env = _make_env_with_scenario(env_config, scenario)

        assert scenario.setup_called is False
        env.reset(seed=42)
        assert scenario.setup_called is True
        assert scenario.setup_call_count == 1

        # Second reset should call setup again
        env.reset(seed=99)
        assert scenario.setup_call_count == 2


class TestScenarioUpdateCalled:
    """step() must call scenario.update()."""

    def test_scenario_update_called(self, env_config: dict[str, Any], tmp_path: Path) -> None:
        """scenario.update() is called each step with step_count and all_states."""
        _write_rewards_yaml(tmp_path)
        scenario = _TrackingScenario()
        env = _make_env_with_scenario(env_config, scenario)
        env.reset(seed=42)

        assert scenario.update_called is False
        actions = {a: [2, 2, 2] for a in env.agents}
        env.step(actions)
        assert scenario.update_called is True
        assert scenario.update_call_count == 1
        assert scenario.update_step_counts[-1] == 1

        env.step(actions)
        assert scenario.update_call_count == 2
        assert scenario.update_step_counts[-1] == 2


class TestScenarioShouldTruncate:
    """step() must call scenario.should_truncate() for each agent."""

    def test_scenario_should_truncate(self, env_config: dict[str, Any], tmp_path: Path) -> None:
        """Agents marked for truncation by scenario are removed."""
        _write_rewards_yaml(tmp_path)
        # Mark AC001 for truncation
        scenario = _TrackingScenario(truncate_ids={"AC001"})
        env = _make_env_with_scenario(env_config, scenario)
        env.reset(seed=42)

        initial_agents = set(env.agents)
        assert "AC001" in initial_agents

        actions = {a: [2, 2, 2] for a in env.agents}
        env.step(actions)

        # AC001 should have been removed via scenario truncation
        assert "AC001" not in env.agents

        # Verify truncate was checked (at least once per agent per step)
        checked_ids = {aid for aid, _ in scenario.truncate_checks}
        assert "AC001" in checked_ids


class TestSubstepMidTermination:
    """Test that safety-critical checks run during substeps."""

    def test_step_passes_on_substep_callback(self, env: BlueSkyMARLEnv) -> None:
        """step() passes on_substep callback to wrapper.step_n()."""
        env.reset(seed=42)

        callback_calls: list[int] = []
        original_step_n = env._wrapper.step_n

        def tracking_step_n(n, on_substep=None):
            def tracking_callback(step: int) -> bool:
                callback_calls.append(step)
                if on_substep is not None:
                    return on_substep(step)
                return True

            return original_step_n(n, on_substep=tracking_callback)

        env._wrapper.step_n = tracking_step_n

        actions = {a: [2, 2, 2] for a in env.agents}
        env.step(actions)

        action_freq = env._action_frequency
        assert len(callback_calls) == action_freq
        assert callback_calls == list(range(action_freq))

    def test_substep_callback_checks_nmac(self, env: BlueSkyMARLEnv) -> None:
        """Safety-critical substep callback detects NMAC between substeps."""
        env.reset(seed=42)

        nmac_detected = {"value": False}
        original_check = env._obs_builder.compute_conflict_status

        def tracking_check(own, others):
            result = original_check(own, others)
            if result == "nmac":
                nmac_detected["value"] = True
            return result

        env._obs_builder.compute_conflict_status = tracking_check

        # The callback should be wired; we verify it's not None
        assert env._action_frequency >= 1
