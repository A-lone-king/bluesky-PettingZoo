"""Tests for AlgorithmFactory (spec4 F2).

Verify that create() returns correct types and from_yaml() loads config.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from bluesky_pettingzoo.training.algorithm_factory import AlgorithmFactory


@pytest.fixture
def dummy_env():
    """Create a simple dummy environment for testing."""

    class DummyEnv(gym.Env):
        observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        def reset(self, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    return DummyEnv()


class TestCreate:
    """AlgorithmFactory.create() should return correct algorithm types."""

    def test_create_ppo(self, dummy_env) -> None:
        """create('PPO') should return PPO instance."""
        from stable_baselines3 import PPO
        algo = AlgorithmFactory.create("PPO", "MlpPolicy", env=dummy_env)
        assert isinstance(algo, PPO)

    def test_create_sac(self, dummy_env) -> None:
        """create('SAC') should return SAC instance."""
        from stable_baselines3 import SAC
        algo = AlgorithmFactory.create("SAC", "MlpPolicy", env=dummy_env)
        assert isinstance(algo, SAC)

    def test_create_td3(self, dummy_env) -> None:
        """create('TD3') should return TD3 instance."""
        from stable_baselines3 import TD3
        algo = AlgorithmFactory.create("TD3", "MlpPolicy", env=dummy_env)
        assert isinstance(algo, TD3)

    def test_create_ddpg(self, dummy_env) -> None:
        """create('DDPG') should return DDPG instance."""
        from stable_baselines3 import DDPG
        algo = AlgorithmFactory.create("DDPG", "MlpPolicy", env=dummy_env)
        assert isinstance(algo, DDPG)

    def test_create_invalid_raises(self) -> None:
        """create('Invalid') should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            AlgorithmFactory.create("Invalid", "MlpPolicy")


class TestFromYaml:
    """AlgorithmFactory.from_yaml() should load config and create algorithm."""

    def test_from_yaml_ppo(self, dummy_env, tmp_path) -> None:
        """from_yaml() should create PPO with config params."""
        import yaml
        config = {
            "PPO": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
            }
        }
        config_path = tmp_path / "algorithms.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        algo = AlgorithmFactory.from_yaml("PPO", "MlpPolicy", config_path, env=dummy_env)
        assert algo.learning_rate == pytest.approx(3e-4)
        assert algo.n_steps == 2048
        assert algo.batch_size == 64

    def test_from_yaml_sac(self, dummy_env, tmp_path) -> None:
        """from_yaml() should create SAC with config params."""
        import yaml
        config = {
            "SAC": {
                "learning_rate": 1e-3,
                "buffer_size": 100000,
                "batch_size": 256,
            }
        }
        config_path = tmp_path / "algorithms.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        algo = AlgorithmFactory.from_yaml("SAC", "MlpPolicy", config_path, env=dummy_env)
        assert algo.learning_rate == pytest.approx(1e-3)
        assert algo.buffer_size == 100000


class TestSupportedAlgorithms:
    """AlgorithmFactory should list supported algorithms."""

    def test_supported_algorithms(self) -> None:
        """supported_algorithms() should return list of algorithm names."""
        algos = AlgorithmFactory.supported_algorithms()
        assert "PPO" in algos
        assert "SAC" in algos
        assert "TD3" in algos
        assert "DDPG" in algos
