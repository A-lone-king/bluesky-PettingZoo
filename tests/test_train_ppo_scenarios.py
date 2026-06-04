"""Tests for PPO multi-scenario training script."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from bluesky_pettingzoo.wrappers.single_agent import SingleAgentGymWrapper

# Import module under test
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from train_ppo_scenarios import SCENARIO_CONFIGS, PPOTrainer, make_scenario_env_factory

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScenarioConfigs:
    """Scenario configuration definitions should be valid."""

    def test_configs_not_empty(self) -> None:
        assert len(SCENARIO_CONFIGS) > 0

    def test_configs_have_required_keys(self) -> None:
        for cfg in SCENARIO_CONFIGS:
            assert "name" in cfg
            assert "scenario" in cfg
            assert "num_aircraft" in cfg

    def test_configs_cover_all_scenarios(self) -> None:
        names = {cfg["name"] for cfg in SCENARIO_CONFIGS}
        expected = {"HorizontalCR", "VerticalCR", "SectorCR", "WaypointNav", "Merge", "Descent"}
        assert expected.issubset(names)


class TestMakeScenarioEnvFactory:
    """make_scenario_env_factory should create valid SingleAgentGymWrapper envs."""

    def test_factory_returns_callable(self, tmp_path: Path) -> None:
        from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario

        scenario = WaypointNavScenario(num_aircraft=2, seed=42)
        factory = make_scenario_env_factory(tmp_path, scenario, num_aircraft=2, max_steps=10)
        assert callable(factory)

    def test_factory_creates_single_agent_wrapper(self, tmp_path: Path) -> None:
        from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario

        scenario = WaypointNavScenario(num_aircraft=2, seed=42)
        factory = make_scenario_env_factory(tmp_path, scenario, num_aircraft=2, max_steps=10)
        env = factory()
        assert isinstance(env, SingleAgentGymWrapper)
        env.close()

    def test_factory_observation_in_space(self, tmp_path: Path) -> None:
        from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario

        scenario = WaypointNavScenario(num_aircraft=2, seed=42)
        factory = make_scenario_env_factory(tmp_path, scenario, num_aircraft=2, max_steps=10)
        env = factory()
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        env.close()


class TestPPOTrainerInit:
    """PPOTrainer should initialize correctly."""

    def test_trainer_creates_model(self, tmp_path: Path) -> None:
        from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario

        scenario = WaypointNavScenario(num_aircraft=2, seed=42)
        trainer = PPOTrainer(
            tmp_path=tmp_path,
            scenario_name="WaypointNav",
            scenario=scenario,
            num_aircraft=2,
            max_steps=10,
            total_timesteps=100,
        )
        assert trainer.model is not None
        trainer.close()

    def test_trainer_has_env(self, tmp_path: Path) -> None:
        from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario

        scenario = WaypointNavScenario(num_aircraft=2, seed=42)
        trainer = PPOTrainer(
            tmp_path=tmp_path,
            scenario_name="WaypointNav",
            scenario=scenario,
            num_aircraft=2,
            max_steps=10,
            total_timesteps=100,
        )
        assert isinstance(trainer.env, SingleAgentGymWrapper)
        trainer.close()


class TestPPOTrainerTrain:
    """PPOTrainer.train() should complete without errors."""

    def test_train_returns_metrics(self, tmp_path: Path) -> None:
        from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario

        scenario = WaypointNavScenario(num_aircraft=2, seed=42)
        trainer = PPOTrainer(
            tmp_path=tmp_path,
            scenario_name="WaypointNav",
            scenario=scenario,
            num_aircraft=2,
            max_steps=10,
            total_timesteps=200,
        )
        metrics = trainer.train()
        assert isinstance(metrics, dict)
        assert "initial_reward" in metrics
        assert "final_reward" in metrics
        trainer.close()


class TestPPOTrainerEvaluate:
    """PPOTrainer.evaluate() should return BaselineMetrics."""

    def test_evaluate_returns_metrics(self, tmp_path: Path) -> None:
        from bluesky_pettingzoo.envs.scenarios.waypoint_nav import WaypointNavScenario
        from scripts.evaluate_baselines import BaselineMetrics

        scenario = WaypointNavScenario(num_aircraft=2, seed=42)
        trainer = PPOTrainer(
            tmp_path=tmp_path,
            scenario_name="WaypointNav",
            scenario=scenario,
            num_aircraft=2,
            max_steps=10,
            total_timesteps=100,
        )
        metrics = trainer.evaluate(n_episodes=2)
        assert isinstance(metrics, BaselineMetrics)
        assert np.isfinite(metrics.mean_reward)
        trainer.close()
