"""Unit tests for MAPPO and IPPO trainers."""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.training.mappo_trainer import (
    IPPOTrainer,
    MAPPOConfig,
    MAPPOEvalResult,
    RayMAPPOAdapter,
    get_mappo_trainer,
)


class TestMAPPOConfig:
    """Tests for MAPPOConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MAPPOConfig()
        assert config.learning_rate == 3e-4
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_range == 0.2
        assert config.ent_coef == 0.01
        assert config.vf_coef == 0.5
        assert config.max_grad_norm == 0.5
        assert config.batch_size == 64
        assert config.n_steps == 2048
        assert config.n_epochs == 10
        assert config.target_kl is None
        assert config.use_centralized_critic is True
        assert config.normalize_advantages is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MAPPOConfig(
            learning_rate=1e-3,
            gamma=0.95,
            batch_size=128,
            n_epochs=5,
        )
        assert config.learning_rate == 1e-3
        assert config.gamma == 0.95
        assert config.batch_size == 128
        assert config.n_epochs == 5


class TestMAPPOEvalResult:
    """Tests for MAPPOEvalResult dataclass."""

    def test_creation(self):
        """Test MAPPOEvalResult creation."""
        result = MAPPOEvalResult(
            scenario="HorizontalCR",
            mean_reward=10.5,
            std_reward=2.3,
            mean_arrival_rate=0.8,
            mean_nmac_rate=0.1,
            mean_steps=50.0,
            num_episodes=20,
        )
        assert result.scenario == "HorizontalCR"
        assert result.mean_reward == 10.5
        assert result.std_reward == 2.3
        assert result.mean_arrival_rate == 0.8
        assert result.mean_nmac_rate == 0.1
        assert result.mean_steps == 50.0
        assert result.num_episodes == 20
        assert result.extended_metrics is None


class TestRayMAPPOAdapter:
    """Tests for RayMAPPOAdapter."""

    def test_availability(self):
        """Test Ray availability check."""
        adapter = RayMAPPOAdapter.__new__(RayMAPPOAdapter)
        adapter._available = True
        assert adapter.available is True

        adapter._available = False
        assert adapter.available is False

    def test_check_availability(self):
        """Test availability check logic."""
        adapter = RayMAPPOAdapter.__new__(RayMAPPOAdapter)
        available = adapter._check_availability()
        assert isinstance(available, bool)


class TestGetMAPPO:
    """Tests for get_mappo_trainer function."""

    def test_returns_trainer(self):
        """Test that get_mappo_trainer returns a valid trainer."""
        try:
            import ray
            trainer = get_mappo_trainer(None)
            assert isinstance(trainer, (IPPOTrainer, RayMAPPOAdapter))
        except ImportError:
            trainer = get_mappo_trainer(None)
            assert isinstance(trainer, IPPOTrainer)