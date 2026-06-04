"""Tests for scenario reward_overrides configuration (spec4 F1).

Verify that scenario YAML reward_overrides correctly merge with base config.
"""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.rewards.calculator import RewardCalculator


@pytest.fixture
def base_config() -> dict:
    return {
        "components": {
            "conflict": {
                "enabled": True,
                "weight": 0.5,
                "nmac_penalty": -1.0,
                "warning_penalty": 0.0,
                "separation_penalty": 0.0,
            },
            "efficiency": {
                "enabled": True,
                "weight": 0.3,
                "arrival_reward": 1.0,
                "step_penalty": 0.0,
            },
            "drift_penalty": {
                "enabled": True,
                "weight": 0.5,
                "scale": -0.1,
            },
        }
    }


class TestScenarioRewardOverrides:
    """Scenario reward_overrides should merge correctly."""

    def test_merge_conflict_weight(self, base_config: dict) -> None:
        """Scenario can override conflict weight."""
        overrides = {
            "conflict": {"weight": 0.8},
        }

        merged = RewardCalculator.merge_reward_config(base_config, overrides)

        assert merged["components"]["conflict"]["weight"] == pytest.approx(0.8)
        # Other values preserved
        assert merged["components"]["conflict"]["nmac_penalty"] == pytest.approx(-1.0)

    def test_merge_drift_scale(self, base_config: dict) -> None:
        """Scenario can override drift penalty scale."""
        overrides = {
            "drift_penalty": {"scale": -0.5},
        }

        merged = RewardCalculator.merge_reward_config(base_config, overrides)

        assert merged["components"]["drift_penalty"]["scale"] == pytest.approx(-0.5)
        assert merged["components"]["drift_penalty"]["weight"] == pytest.approx(0.5)

    def test_merge_multiple_components(self, base_config: dict) -> None:
        """Scenario can override multiple components."""
        overrides = {
            "conflict": {"weight": 0.9},
            "efficiency": {"arrival_reward": 5.0},
        }

        merged = RewardCalculator.merge_reward_config(base_config, overrides)

        assert merged["components"]["conflict"]["weight"] == pytest.approx(0.9)
        assert merged["components"]["efficiency"]["arrival_reward"] == pytest.approx(5.0)
        # Unchanged values preserved
        assert merged["components"]["drift_penalty"]["scale"] == pytest.approx(-0.1)

    def test_no_overrides_preserves_base(self, base_config: dict) -> None:
        """Empty overrides should preserve all base config values."""
        merged = RewardCalculator.merge_reward_config(base_config, {})

        assert merged["components"]["conflict"]["nmac_penalty"] == pytest.approx(-1.0)
        assert merged["components"]["efficiency"]["arrival_reward"] == pytest.approx(1.0)
        assert merged["components"]["drift_penalty"]["scale"] == pytest.approx(-0.1)

    def test_merge_does_not_mutate_base(self, base_config: dict) -> None:
        """Merge should not mutate the original base config."""
        overrides = {
            "conflict": {"nmac_penalty": -5.0},
        }
        original_penalty = base_config["components"]["conflict"]["nmac_penalty"]

        RewardCalculator.merge_reward_config(base_config, overrides)

        assert base_config["components"]["conflict"]["nmac_penalty"] == (
            pytest.approx(original_penalty)
        )
