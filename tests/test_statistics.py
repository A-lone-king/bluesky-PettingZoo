"""Unit tests for statistical significance testing."""

from __future__ import annotations

import numpy as np
import pytest

from bluesky_pettingzoo.training.statistics import (
    StatisticalTest,
    cohen_d,
    mean_confidence_interval,
    wilcoxon_rank_sum_test,
)


class TestCohenD:
    """Tests for Cohen's d effect size."""

    def test_identical_groups(self):
        """Cohen's d should be 0 for identical groups."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohen_d(a, a)
        assert abs(d) < 1e-10

    def test_large_effect(self):
        """Two well-separated groups should give a large effect."""
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohen_d(a, b)
        assert d > 0.8  # large effect

    def test_sign_direction(self):
        """Positive d when group a > group b."""
        a = np.array([5.0, 6.0, 7.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cohen_d(a, b) > 0
        assert cohen_d(b, a) < 0

    def test_empty_group(self):
        """Should raise ValueError for empty input."""
        with pytest.raises(ValueError):
            cohen_d(np.array([]), np.array([1.0]))


class TestMeanConfidenceInterval:
    """Tests for 95% confidence interval of the mean."""

    def test_basic_ci(self):
        """CI should contain the mean."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        mean, lower, upper = mean_confidence_interval(data)
        assert lower < mean < upper

    def test_single_element(self):
        """Single element: CI should be the value itself (no variance)."""
        mean, lower, upper = mean_confidence_interval(np.array([42.0]))
        assert mean == 42.0
        assert lower == 42.0
        assert upper == 42.0

    def test_custom_confidence(self):
        """Test with 99% confidence level."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 4)
        _, lower_95, upper_95 = mean_confidence_interval(data, confidence=0.95)
        _, lower_99, upper_99 = mean_confidence_interval(data, confidence=0.99)
        # 99% CI should be wider
        assert (upper_99 - lower_99) >= (upper_95 - lower_95)

    def test_empty_data(self):
        """Should raise ValueError for empty data."""
        with pytest.raises(ValueError):
            mean_confidence_interval(np.array([]))


class TestWilcoxonRankSumTest:
    """Tests for Wilcoxon rank-sum (Mann-Whitney U) test."""

    def test_identical_distributions(self):
        """p-value should be high for identical distributions."""
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 50)
        b = rng.normal(0, 1, 50)
        _, p = wilcoxon_rank_sum_test(a, b)
        assert p > 0.05  # not significant

    def test_different_distributions(self):
        """p-value should be low for clearly different distributions."""
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 50)
        b = rng.normal(5, 1, 50)
        _, p = wilcoxon_rank_sum_test(a, b)
        assert p < 0.05  # significant

    def test_small_samples(self):
        """Should handle small samples gracefully."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 11.0, 12.0])
        stat, p = wilcoxon_rank_sum_test(a, b)
        assert 0.0 <= p <= 1.0
        assert isinstance(stat, float)


class TestStatisticalTestWrapper:
    """Tests for the StatisticalTest convenience class."""

    def test_compare_two_groups(self):
        """Compare two groups and get full results."""
        rng = np.random.RandomState(42)
        a = rng.normal(10, 2, 30)
        b = rng.normal(15, 2, 30)
        result = StatisticalTest.compare(a, b)
        assert "cohen_d" in result
        assert "p_value" in result
        assert "mean_a" in result
        assert "mean_b" in result
        assert "ci_a" in result
        assert "ci_b" in result
        assert "significant" in result

    def test_significance_flag(self):
        """Significant flag should be True for different groups."""
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 100)
        b = rng.normal(5, 1, 100)
        result = StatisticalTest.compare(a, b, alpha=0.05)
        assert result["significant"] is True

    def test_custom_alpha(self):
        """Custom alpha threshold."""
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 50)
        b = rng.normal(0.5, 1, 50)
        # At alpha=0.05, might be significant; at alpha=0.01, might not
        result_strict = StatisticalTest.compare(a, b, alpha=0.01)
        assert "significant" in result_strict
