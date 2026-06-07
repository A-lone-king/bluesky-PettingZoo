"""Tests for action space ablation experiments (action-validation-001)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from bluesky_pettingzoo.training.ablation import (
    AblationReporter,
    AblationResult,
    AblationRunner,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_config() -> dict[str, Any]:
    """Create a test configuration."""
    return {
        "experiments": {
            "test_discrete": {
                "name": "Test Discrete",
                "action_type": "discrete",
                "action_dims": [5, 5, 5],
                "action_labels": ["heading", "altitude", "speed"],
                "description": "Test discrete experiment",
            },
            "test_continuous": {
                "name": "Test Continuous",
                "action_type": "continuous",
                "action_dims": 3,
                "action_labels": ["heading", "altitude", "speed"],
                "description": "Test continuous experiment",
            },
        },
        "training": {
            "total_timesteps": 1000,
            "eval_episodes": 5,
        },
    }


def _make_mock_env_factory() -> Any:  # noqa: ANN401
    """Create a mock environment factory."""
    from gymnasium import spaces

    def factory(action_config: dict[str, Any]) -> MagicMock:
        env = MagicMock()
        env.possible_agents = ["agent_0", "agent_1"]
        env.agents = ["agent_0", "agent_1"]

        action_type = action_config.get("action", {}).get("type", "discrete")
        action_dims = action_config.get("action", {}).get("dims", [5, 5, 5])

        if action_type == "discrete":
            env.action_space.return_value = spaces.MultiDiscrete(action_dims)
        else:
            env.action_space.return_value = spaces.Box(
                low=-1.0, high=1.0, shape=(action_dims,), dtype=np.float32
            )

        def mock_reset(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            obs = {}
            infos = {}
            for agent in env.possible_agents:
                obs[agent] = np.random.randn(9).astype(np.float32)
                infos[agent] = {}
            return obs, infos

        def mock_step(
            actions: dict[str, Any],
        ) -> tuple[
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
        ]:
            obs = {}
            rewards = {}
            terminations = {}
            truncations = {}
            infos = {}
            for agent in env.possible_agents:
                obs[agent] = np.random.randn(9).astype(np.float32)
                rewards[agent] = np.random.randn()
                terminations[agent] = False
                truncations[agent] = False
                infos[agent] = {"conflict_status": "safe"}
            return obs, rewards, terminations, truncations, infos

        env.reset.side_effect = mock_reset
        env.step.side_effect = mock_step
        return env

    return factory


def _make_mock_agent_factory() -> Any:  # noqa: ANN401
    """Create a mock agent factory."""

    def factory(env: Any) -> MagicMock:  # noqa: ANN401
        agent = MagicMock()
        action_space = env.action_space()

        def predict(obs: Any) -> Any:  # noqa: ANN401
            return action_space.sample()

        agent.predict.side_effect = predict
        return agent

    return factory


# ---------------------------------------------------------------------------
# AblationResult Tests
# ---------------------------------------------------------------------------


class TestAblationResult:
    """Tests for AblationResult dataclass."""

    def test_creation(self) -> None:
        """Test AblationResult creation."""
        result = AblationResult(
            experiment_id="test",
            name="Test Experiment",
            action_type="discrete",
            action_dims=[5, 5, 5],
            action_labels=["heading", "altitude", "speed"],
            total_timesteps=1000,
        )
        assert result.experiment_id == "test"
        assert result.name == "Test Experiment"
        assert result.action_type == "discrete"
        assert result.action_dims == [5, 5, 5]

    def test_mean_reward(self) -> None:
        """Test mean_reward calculation."""
        result = AblationResult(
            experiment_id="test",
            name="Test",
            action_type="discrete",
            action_dims=[5],
            action_labels=["heading"],
            total_timesteps=1000,
            episode_rewards=[1.0, 2.0, 3.0],
        )
        assert result.mean_reward == pytest.approx(2.0)

    def test_std_reward(self) -> None:
        """Test std_reward calculation."""
        result = AblationResult(
            experiment_id="test",
            name="Test",
            action_type="discrete",
            action_dims=[5],
            action_labels=["heading"],
            total_timesteps=1000,
            episode_rewards=[1.0, 2.0, 3.0],
        )
        assert result.std_reward == pytest.approx(0.816, abs=0.01)

    def test_empty_rewards(self) -> None:
        """Test with empty rewards."""
        result = AblationResult(
            experiment_id="test",
            name="Test",
            action_type="discrete",
            action_dims=[5],
            action_labels=["heading"],
            total_timesteps=1000,
        )
        assert result.mean_reward == 0.0
        assert result.std_reward == 0.0

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        result = AblationResult(
            experiment_id="test",
            name="Test",
            action_type="discrete",
            action_dims=[5],
            action_labels=["heading"],
            total_timesteps=1000,
            episode_rewards=[1.0, 2.0],
        )
        d = result.to_dict()
        assert d["experiment_id"] == "test"
        assert d["mean_reward"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# AblationRunner Tests
# ---------------------------------------------------------------------------


class TestAblationRunner:
    """Tests for AblationRunner."""

    def test_list_experiments(self) -> None:
        """Test listing experiments."""
        config = _make_test_config()
        runner = AblationRunner(config)

        experiments = runner.list_experiments()
        assert len(experiments) == 2
        assert "test_discrete" in experiments
        assert "test_continuous" in experiments

    def test_list_discrete_experiments(self) -> None:
        """Test listing discrete experiments."""
        config = _make_test_config()
        runner = AblationRunner(config)

        discrete = runner.list_discrete_experiments()
        assert len(discrete) == 1
        assert "test_discrete" in discrete

    def test_list_continuous_experiments(self) -> None:
        """Test listing continuous experiments."""
        config = _make_test_config()
        runner = AblationRunner(config)

        continuous = runner.list_continuous_experiments()
        assert len(continuous) == 1
        assert "test_continuous" in continuous

    def test_get_experiment_config(self) -> None:
        """Test getting experiment config."""
        config = _make_test_config()
        runner = AblationRunner(config)

        exp_cfg = runner.get_experiment_config("test_discrete")
        assert exp_cfg is not None
        assert exp_cfg["name"] == "Test Discrete"
        assert exp_cfg["action_type"] == "discrete"

    def test_get_unknown_experiment(self) -> None:
        """Test getting unknown experiment."""
        config = _make_test_config()
        runner = AblationRunner(config)

        exp_cfg = runner.get_experiment_config("unknown")
        assert exp_cfg is None

    def test_create_discrete_action_config(self) -> None:
        """Test creating discrete action config."""
        config = _make_test_config()
        runner = AblationRunner(config)

        action_config = runner.create_action_space_config("test_discrete")
        assert action_config["action"]["type"] == "discrete"
        assert action_config["action"]["dims"] == [5, 5, 5]

    def test_create_continuous_action_config(self) -> None:
        """Test creating continuous action config."""
        config = _make_test_config()
        runner = AblationRunner(config)

        action_config = runner.create_action_space_config("test_continuous")
        assert action_config["action"]["type"] == "continuous"
        assert action_config["action"]["dims"] == 3

    def test_create_unknown_action_config(self) -> None:
        """Test creating config for unknown experiment."""
        config = _make_test_config()
        runner = AblationRunner(config)

        with pytest.raises(ValueError, match="Unknown experiment"):
            runner.create_action_space_config("unknown")

    def test_run_experiment(self) -> None:
        """Test running an experiment."""
        config = _make_test_config()
        runner = AblationRunner(config)

        env_factory = _make_mock_env_factory()
        agent_factory = _make_mock_agent_factory()

        result = runner.run_experiment(
            experiment_id="test_discrete",
            env_factory=env_factory,
            agent_factory=agent_factory,
            num_episodes=3,
            max_steps_per_episode=50,
        )

        assert result.experiment_id == "test_discrete"
        assert result.name == "Test Discrete"
        assert len(result.episode_rewards) == 3
        assert len(result.episode_lengths) == 3
        assert result.training_time_seconds > 0


# ---------------------------------------------------------------------------
# AblationReporter Tests
# ---------------------------------------------------------------------------


class TestAblationReporter:
    """Tests for AblationReporter."""

    def test_generate_report(self, tmp_path: Any) -> None:
        """Test generating a report."""
        reporter = AblationReporter(str(tmp_path))

        results = [
            AblationResult(
                experiment_id="test1",
                name="Test 1",
                action_type="discrete",
                action_dims=[5],
                action_labels=["heading"],
                total_timesteps=1000,
                episode_rewards=[1.0, 2.0, 3.0],
            ),
            AblationResult(
                experiment_id="test2",
                name="Test 2",
                action_type="continuous",
                action_dims=1,
                action_labels=["heading"],
                total_timesteps=1000,
                episode_rewards=[1.5, 2.5, 3.5],
            ),
        ]

        report_path = reporter.generate_report(results)
        assert report_path.endswith("ablation_report.md")

        # Check report content
        with open(report_path, encoding="utf-8") as f:
            content = f.read()

        assert "Action Space Ablation Report" in content
        assert "Test 1" in content
        assert "Test 2" in content
        assert "Summary" in content

    def test_save_results(self, tmp_path: Any) -> None:
        """Test saving results to JSON."""
        reporter = AblationReporter(str(tmp_path))

        results = [
            AblationResult(
                experiment_id="test1",
                name="Test 1",
                action_type="discrete",
                action_dims=[5],
                action_labels=["heading"],
                total_timesteps=1000,
                episode_rewards=[1.0, 2.0],
            ),
        ]

        results_path = reporter.save_results(results)
        assert results_path.endswith("ablation_results.json")

        # Check JSON content
        import json

        with open(results_path, encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["experiment_id"] == "test1"


# ---------------------------------------------------------------------------
# Config Loading Tests
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Tests for loading ablation config."""

    def test_load_config(self) -> None:
        """Test loading config from YAML file."""
        config_path = "config/ablation_experiments.yaml"
        try:
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            assert "experiments" in config
            assert len(config["experiments"]) > 0
        except FileNotFoundError:
            pytest.skip("Config file not found")

    def test_config_has_required_fields(self) -> None:
        """Test that config has required fields."""
        config_path = "config/ablation_experiments.yaml"
        try:
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            for exp_id, exp_cfg in config["experiments"].items():
                assert "name" in exp_cfg, f"Missing 'name' in {exp_id}"
                assert "action_type" in exp_cfg, f"Missing 'action_type' in {exp_id}"
                assert "action_dims" in exp_cfg, f"Missing 'action_dims' in {exp_id}"
                assert "action_labels" in exp_cfg, f"Missing 'action_labels' in {exp_id}"
        except FileNotFoundError:
            pytest.skip("Config file not found")
