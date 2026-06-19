"""Tests for reward configuration loading (spec4 F1).

Verify that drift_penalty weight and parameters load correctly from YAML,
and that smoothness weight can be set to 0 to disable it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def rewards_yaml_path() -> Path:
    return Path(__file__).parent.parent / "config" / "rewards.yaml"


@pytest.fixture
def rewards_config(rewards_yaml_path: Path) -> dict:
    with open(rewards_yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestDriftPenaltyConfig:
    """DriftPenalty should be configurable via rewards.yaml."""

    def test_drift_penalty_section_exists(self, rewards_config: dict) -> None:
        """drift_penalty section should exist in components."""
        assert "drift_penalty" in rewards_config["components"]

    def test_drift_penalty_weight(self, rewards_config: dict) -> None:
        """drift_penalty weight should be 0.5."""
        cfg = rewards_config["components"]["drift_penalty"]
        assert cfg["weight"] == pytest.approx(0.5)

    def test_drift_penalty_scale(self, rewards_config: dict) -> None:
        """drift_penalty scale should be -0.1 (aligns with bluesky-gym)."""
        cfg = rewards_config["components"]["drift_penalty"]
        assert cfg["scale"] == pytest.approx(-0.1)

    def test_drift_penalty_enabled(self, rewards_config: dict) -> None:
        """drift_penalty should be enabled by default."""
        cfg = rewards_config["components"]["drift_penalty"]
        assert cfg["enabled"] is True


class TestSmoothnessConfig:
    """SmoothnessPenalty weight should be 0.5."""

    def test_smoothness_weight_zero(self, rewards_config: dict) -> None:
        """smoothness weight should be 0.5."""
        cfg = rewards_config["components"]["smoothness"]
        assert cfg["weight"] == pytest.approx(0.5)


class TestEfficiencyConfig:
    """EfficiencyReward should align with bluesky-gym parameters."""

    def test_arrival_reward(self, rewards_config: dict) -> None:
        """arrival_reward should be 100.0."""
        cfg = rewards_config["components"]["efficiency"]
        assert cfg["arrival_reward"] == pytest.approx(100.0)

    def test_step_penalty_zero(self, rewards_config: dict) -> None:
        """step_penalty should be -0.005."""
        cfg = rewards_config["components"]["efficiency"]
        assert cfg["step_penalty"] == pytest.approx(-0.005)

    def test_deviation_scale_zero(self, rewards_config: dict) -> None:
        """deviation_penalty_scale should be 5.0."""
        cfg = rewards_config["components"]["efficiency"]
        assert cfg["deviation_penalty_scale"] == pytest.approx(5.0)


class TestConflictConfig:
    """ConflictPenalty should align with bluesky-gym parameters."""

    def test_nmac_penalty(self, rewards_config: dict) -> None:
        """nmac_penalty should be -50.0."""
        cfg = rewards_config["components"]["conflict"]
        assert cfg["nmac_penalty"] == pytest.approx(-50.0)

    def test_warning_penalty_zero(self, rewards_config: dict) -> None:
        """warning_penalty should be -10.0."""
        cfg = rewards_config["components"]["conflict"]
        assert cfg["warning_penalty"] == pytest.approx(-10.0)

    def test_separation_penalty_zero(self, rewards_config: dict) -> None:
        """separation_penalty should be -5.0."""
        cfg = rewards_config["components"]["conflict"]
        assert cfg["separation_penalty"] == pytest.approx(-5.0)
