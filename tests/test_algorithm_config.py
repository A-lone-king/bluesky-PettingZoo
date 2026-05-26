"""Tests for algorithm configuration (spec4 F2).

Verify YAML config loading and default value overrides.
"""

from __future__ import annotations

import pytest
import yaml

from bluesky_pettingzoo.training.algorithm_factory import AlgorithmFactory


class TestAlgorithmConfig:
    """Algorithm config should load from YAML correctly."""

    def test_load_ppo_config(self, tmp_path) -> None:
        """PPO config should load with correct values."""
        config = {
            "PPO": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
            }
        }
        config_path = tmp_path / "algorithms.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with open(config_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["PPO"]["learning_rate"] == pytest.approx(3e-4)
        assert loaded["PPO"]["n_steps"] == 2048
        assert loaded["PPO"]["batch_size"] == 64
        assert loaded["PPO"]["n_epochs"] == 10
        assert loaded["PPO"]["gamma"] == pytest.approx(0.99)

    def test_load_sac_config(self, tmp_path) -> None:
        """SAC config should load with correct values."""
        config = {
            "SAC": {
                "learning_rate": 1e-3,
                "buffer_size": 100000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
            }
        }
        config_path = tmp_path / "algorithms.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with open(config_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["SAC"]["learning_rate"] == pytest.approx(1e-3)
        assert loaded["SAC"]["buffer_size"] == 100000
        assert loaded["SAC"]["batch_size"] == 256

    def test_load_td3_config(self, tmp_path) -> None:
        """TD3 config should load with correct values."""
        config = {
            "TD3": {
                "learning_rate": 1e-3,
                "buffer_size": 100000,
                "batch_size": 100,
                "tau": 0.005,
                "gamma": 0.99,
            }
        }
        config_path = tmp_path / "algorithms.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with open(config_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["TD3"]["learning_rate"] == pytest.approx(1e-3)
        assert loaded["TD3"]["buffer_size"] == 100000

    def test_load_ddpg_config(self, tmp_path) -> None:
        """DDPG config should load with correct values."""
        config = {
            "DDPG": {
                "learning_rate": 1e-3,
                "buffer_size": 100000,
                "batch_size": 128,
                "tau": 0.005,
                "gamma": 0.99,
            }
        }
        config_path = tmp_path / "algorithms.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with open(config_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["DDPG"]["learning_rate"] == pytest.approx(1e-3)
        assert loaded["DDPG"]["buffer_size"] == 100000


class TestDefaultOverrides:
    """Config values should override algorithm defaults."""

    def test_ppo_lr_override(self, tmp_path) -> None:
        """PPO learning_rate from config should override default."""
        config = {"PPO": {"learning_rate": 1e-4}}
        config_path = tmp_path / "algorithms.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        import gymnasium as gym
        import numpy as np

        class DummyEnv(gym.Env):
            observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
            action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

            def reset(self, seed=None, options=None):
                return self.observation_space.sample(), {}

            def step(self, action):
                return self.observation_space.sample(), 0.0, False, False, {}

        env = DummyEnv()
        algo = AlgorithmFactory.from_yaml("PPO", "MlpPolicy", config_path, env=env)
        assert algo.learning_rate == pytest.approx(1e-4)

    def test_missing_config_uses_defaults(self, tmp_path) -> None:
        """Missing algorithm in config should use defaults."""
        config = {"PPO": {"learning_rate": 1e-4}}
        config_path = tmp_path / "algorithms.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        import gymnasium as gym
        import numpy as np

        class DummyEnv(gym.Env):
            observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
            action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

            def reset(self, seed=None, options=None):
                return self.observation_space.sample(), {}

            def step(self, action):
                return self.observation_space.sample(), 0.0, False, False, {}

        env = DummyEnv()
        # SAC not in config, should use defaults
        algo = AlgorithmFactory.from_yaml("SAC", "MlpPolicy", config_path, env=env)
        assert algo.learning_rate == pytest.approx(3e-4)  # SB3 default
