"""Tests for DriftPenalty registration in reward components."""

from __future__ import annotations


class TestDriftPenaltyRegistration:
    """Verify DriftPenalty is properly registered and importable."""

    def test_drift_penalty_importable_from_components(self):
        from bluesky_pettingzoo.rewards.components import DriftPenalty

        assert DriftPenalty is not None

    def test_drift_penalty_in_all(self):
        from bluesky_pettingzoo.rewards.components import __all__

        assert "DriftPenalty" in __all__

    def test_drift_penalty_is_reward_component(self):
        from bluesky_pettingzoo.rewards.base import RewardComponent
        from bluesky_pettingzoo.rewards.components import DriftPenalty

        assert issubclass(DriftPenalty, RewardComponent)
