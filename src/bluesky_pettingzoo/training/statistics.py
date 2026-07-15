"""Statistical significance testing for paper-level experiments.

Provides Wilcoxon rank-sum test, Cohen's d effect size, and 95% confidence
intervals for comparing RL algorithm performance across seeds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def cohen_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups.

    Cohen's d measures the standardized difference between two means:
        d = (mean_a - mean_b) / pooled_std

    Interpretation:
        |d| < 0.2  — negligible
        |d| < 0.5  — small
        |d| < 0.8  — medium
        |d| >= 0.8 — large

    Args:
        group_a: First sample array.
        group_b: Second sample array.

    Returns:
        Cohen's d effect size (positive when group_a > group_b).

    Raises:
        ValueError: If either group is empty.
    """
    if len(group_a) == 0 or len(group_b) == 0:
        raise ValueError("Both groups must be non-empty.")

    mean_a = float(np.mean(group_a))
    mean_b = float(np.mean(group_b))

    var_a = float(np.var(group_a, ddof=1)) if len(group_a) > 1 else 0.0
    var_b = float(np.var(group_b, ddof=1)) if len(group_b) > 1 else 0.0
    n_a, n_b = len(group_a), len(group_b)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std == 0:
        return 0.0

    return (mean_a - mean_b) / pooled_std


def mean_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute confidence interval for the mean using t-distribution.

    Args:
        data: 1-D sample array.
        confidence: Confidence level (e.g. 0.95 for 95% CI).

    Returns:
        Tuple of (mean, lower_bound, upper_bound).

    Raises:
        ValueError: If data is empty.
    """
    if len(data) == 0:
        raise ValueError("Data must be non-empty.")

    n = len(data)
    mean = float(np.mean(data))

    if n == 1:
        return mean, mean, mean

    sem = float(stats.sem(data))  # standard error of the mean
    alpha = 1 - confidence
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    margin = t_crit * sem

    return mean, mean - margin, mean + margin


def wilcoxon_rank_sum_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> tuple[float, float]:
    """Perform Wilcoxon rank-sum (Mann-Whitney U) test.

    Non-parametric test for whether two independent samples come from the
    same distribution. Does not assume normality.

    Args:
        group_a: First sample array.
        group_b: Second sample array.

    Returns:
        Tuple of (statistic, p_value).
    """
    result = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


class StatisticalTest:
    """Convenience wrapper for comparing two algorithm results."""

    @staticmethod
    def compare(
        group_a: np.ndarray,
        group_b: np.ndarray,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Run full statistical comparison between two groups.

        Args:
            group_a: Results from algorithm A (e.g. rewards across seeds).
            group_b: Results from algorithm B.
            alpha: Significance level (default 0.05).

        Returns:
            Dict with keys: cohen_d, p_value, mean_a, mean_b,
                ci_a (lower, upper), ci_b (lower, upper), significant.
        """
        _, p_value = wilcoxon_rank_sum_test(group_a, group_b)
        d = cohen_d(group_a, group_b)
        mean_a, ci_a_lower, ci_a_upper = mean_confidence_interval(group_a)
        mean_b, ci_b_lower, ci_b_upper = mean_confidence_interval(group_b)

        return {
            "cohen_d": d,
            "p_value": p_value,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "ci_a": (ci_a_lower, ci_a_upper),
            "ci_b": (ci_b_lower, ci_b_upper),
            "significant": p_value < alpha,
        }
