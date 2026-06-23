"""Unit tests for notebooks/statistics/stats_utils.py.

Tests verify correctness of statistical functions using known-outcome
datasets. All tests are deterministic (seeded RNGs) and self-contained
(no external data dependencies).

Run:
    pytest tests/test_stats_utils.py -v
"""

from __future__ import annotations

# Ensure the project root is importable
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notebooks.statistics.stats_utils import (
    analytical_ci,
    apply_correction,
    bootstrap_ci,
    check_independence_note,
    check_normality,
    check_variance_homogeneity,
    cohens_d,
    compute_vif,
    epsilon_squared,
    eta_squared,
    format_ci_table,
    format_test_result,
    multi_group_test,
    ols_regression,
    paired_test,
    rank_biserial_correlation,
    two_group_test,
)

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture()
def normal_data() -> np.ndarray:
    """Generate normally distributed data."""
    return np.random.default_rng(42).normal(loc=100, scale=15, size=500)


@pytest.fixture()
def skewed_data() -> np.ndarray:
    """Generate right-skewed (exponential) data — mimics Airbnb prices."""
    return np.random.default_rng(42).exponential(scale=100, size=500)


@pytest.fixture()
def two_distinct_groups() -> tuple[np.ndarray, np.ndarray]:
    """Two clearly separated groups for testing significance."""
    rng = np.random.default_rng(42)
    a = rng.normal(loc=50, scale=10, size=200)
    b = rng.normal(loc=80, scale=10, size=200)
    return a, b


@pytest.fixture()
def two_similar_groups() -> tuple[np.ndarray, np.ndarray]:
    """Two groups drawn from the same distribution."""
    rng = np.random.default_rng(42)
    a = rng.normal(loc=50, scale=10, size=200)
    b = rng.normal(loc=50, scale=10, size=200)
    return a, b


# ===================================================================
# Assumption Checks
# ===================================================================


class TestCheckNormality:
    """Tests for check_normality function."""

    def test_normal_data_passes(self, normal_data: np.ndarray):
        result = check_normality(normal_data, alpha=0.05)
        assert result.is_satisfied
        assert result.test_name == "Shapiro-Wilk"
        assert result.sample_size == 500

    def test_skewed_data_fails(self, skewed_data: np.ndarray):
        result = check_normality(skewed_data, alpha=0.05)
        assert not result.is_satisfied

    def test_large_sample_uses_dagostino(self):
        rng = np.random.default_rng(42)
        large = rng.normal(0, 1, size=6000)
        result = check_normality(large, max_sample_for_shapiro=5000)
        assert result.test_name == "D'Agostino-Pearson"
        assert result.sample_size == 6000

    def test_insufficient_data(self):
        result = check_normality(np.array([1, 2, 3]))
        assert not result.is_satisfied
        assert "Insufficient" in result.note

    def test_handles_nan(self, normal_data: np.ndarray):
        data_with_nan = np.append(normal_data, [np.nan, np.nan, np.nan])
        result = check_normality(data_with_nan)
        assert result.sample_size == 500  # NaNs excluded


class TestCheckVarianceHomogeneity:
    """Tests for check_variance_homogeneity function."""

    def test_equal_variance_passes(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 10, size=200)
        b = rng.normal(5, 10, size=200)  # same variance
        result = check_variance_homogeneity(a, b)
        assert result.is_satisfied

    def test_unequal_variance_fails(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 5, size=200)
        b = rng.normal(0, 50, size=200)  # 10x variance
        result = check_variance_homogeneity(a, b)
        assert not result.is_satisfied

    def test_small_group_handled(self):
        result = check_variance_homogeneity(np.array([1]), np.array([2, 3]))
        assert not result.is_satisfied


class TestCheckIndependenceNote:
    """Tests for check_independence_note function."""

    def test_returns_documentation(self):
        result = check_independence_note("Each listing is independent by design.")
        assert result.is_satisfied
        assert "independent" in result.note


# ===================================================================
# Effect Sizes
# ===================================================================


class TestCohensD:
    """Tests for cohens_d function."""

    def test_large_effect(self, two_distinct_groups):
        a, b = two_distinct_groups
        d, mag = cohens_d(a, b)
        assert d < -2.0  # a < b by ~30 units with sd=10
        assert mag == "large"

    def test_negligible_effect(self, two_similar_groups):
        a, b = two_similar_groups
        d, mag = cohens_d(a, b)
        assert abs(d) < 0.3

    def test_zero_variance(self):
        a = np.array([5, 5, 5, 5])
        b = np.array([5, 5, 5, 5])
        d, mag = cohens_d(a, b)
        assert d == 0.0
        assert mag == "negligible"

    def test_handles_nan(self):
        a = np.array([1, 2, 3, np.nan, 5])
        b = np.array([10, 11, 12, 13, 14])
        d, mag = cohens_d(a, b)
        assert d < 0  # a is clearly smaller


class TestRankBiserialCorrelation:
    """Tests for rank_biserial_correlation function."""

    def test_known_values(self):
        # U = 0 means perfect separation → r should be near 1
        r, mag = rank_biserial_correlation(u_statistic=0, n1=10, n2=10)
        assert r == 1.0
        assert mag == "large"

    def test_no_difference(self):
        r, mag = rank_biserial_correlation(u_statistic=50, n1=10, n2=10)
        assert r == 0.0
        assert mag == "negligible"

    def test_zero_n(self):
        r, _ = rank_biserial_correlation(u_statistic=10, n1=0, n2=5)
        assert r == 0.0


class TestEtaSquared:
    """Tests for eta_squared function."""

    def test_zero_f(self):
        es, mag = eta_squared(0.0, 3, 100)
        assert es == 0.0

    def test_large_f(self):
        es, mag = eta_squared(50.0, 3, 100)
        assert es > 0.5
        assert mag == "large"


class TestEpsilonSquared:
    """Tests for epsilon_squared function."""

    def test_basic(self):
        es, mag = epsilon_squared(h_statistic=30.0, n=500, k=5)
        assert es > 0
        assert isinstance(mag, str)


# ===================================================================
# Confidence Intervals
# ===================================================================


class TestBootstrapCI:
    """Tests for bootstrap_ci function."""

    def test_covers_true_mean(self, normal_data: np.ndarray):
        ci = bootstrap_ci(normal_data, ci_level=0.95)
        assert ci.ci_lower < 100 < ci.ci_upper  # true mean = 100
        assert ci.n == 500
        assert ci.method == "bootstrap"

    def test_narrow_for_large_n(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50, 1, size=10000)
        ci = bootstrap_ci(data)
        assert ci.ci_upper - ci.ci_lower < 0.1  # very narrow

    def test_empty_data(self):
        ci = bootstrap_ci(np.array([]))
        assert np.isnan(ci.mean)

    def test_handles_nan(self, normal_data: np.ndarray):
        data = np.append(normal_data, [np.nan] * 10)
        ci = bootstrap_ci(data)
        assert ci.n == 500


class TestAnalyticalCI:
    """Tests for analytical_ci function."""

    def test_covers_true_mean(self, normal_data: np.ndarray):
        ci = analytical_ci(normal_data, ci_level=0.95)
        assert ci.ci_lower < 100 < ci.ci_upper

    def test_single_observation(self):
        ci = analytical_ci(np.array([5.0]))
        assert np.isnan(ci.mean)


# ===================================================================
# Hypothesis Test Wrappers
# ===================================================================


class TestTwoGroupTest:
    """Tests for two_group_test function."""

    def test_significant_difference(self, two_distinct_groups):
        a, b = two_distinct_groups
        result = two_group_test(
            a,
            b,
            hypothesis_id="H_test",
            null_hypothesis="Medians are equal",
            alt_hypothesis="Medians differ",
            group_a_label="Low",
            group_b_label="High",
        )
        assert result.is_significant
        assert result.p_value < 0.001
        assert result.effect_magnitude == "large"
        assert len(result.assumptions_checked) >= 2

    def test_no_significant_difference(self, two_similar_groups):
        a, b = two_similar_groups
        result = two_group_test(
            a,
            b,
            hypothesis_id="H_null",
            null_hypothesis="Medians are equal",
            alt_hypothesis="Medians differ",
        )
        # May or may not be significant with random data, but effect should be small
        assert result.effect_magnitude in ("negligible", "small")

    def test_skewed_data_uses_mannwhitney(self, skewed_data: np.ndarray):
        a = skewed_data[:250]
        b = skewed_data[250:] + 50
        result = two_group_test(
            a,
            b,
            hypothesis_id="H_skew",
            null_hypothesis="Medians are equal",
            alt_hypothesis="Medians differ",
        )
        assert result.test_name == "Mann-Whitney U"

    def test_one_sided(self, two_distinct_groups):
        a, b = two_distinct_groups
        result = two_group_test(
            b,
            a,
            hypothesis_id="H_one",
            null_hypothesis="B ≤ A",
            alt_hypothesis="B > A",
            alternative="greater",
        )
        assert result.is_significant


class TestMultiGroupTest:
    """Tests for multi_group_test function."""

    def test_significant_groups(self):
        rng = np.random.default_rng(42)
        groups = {
            "Low": rng.normal(10, 2, 100),
            "Mid": rng.normal(20, 2, 100),
            "High": rng.normal(30, 2, 100),
        }
        result = multi_group_test(
            groups,
            hypothesis_id="H_multi",
            null_hypothesis="All group medians equal",
            alt_hypothesis="At least one differs",
        )
        assert result.is_significant
        assert result.posthoc_results is not None
        assert len(result.posthoc_results) == 3  # 3 choose 2

    def test_no_difference(self):
        rng = np.random.default_rng(42)
        groups = {
            "A": rng.normal(50, 10, 100),
            "B": rng.normal(50, 10, 100),
            "C": rng.normal(50, 10, 100),
        }
        result = multi_group_test(
            groups,
            hypothesis_id="H_null",
            null_hypothesis="All group medians equal",
            alt_hypothesis="At least one differs",
        )
        assert result.effect_magnitude in ("negligible", "small")


class TestPairedTest:
    """Tests for paired_test function."""

    def test_significant_paired(self):
        rng = np.random.default_rng(42)
        a = rng.normal(100, 10, 200)
        b = a + rng.normal(5, 2, 200)  # systematic shift
        result = paired_test(
            b,
            a,
            hypothesis_id="H_paired",
            null_hypothesis="No difference",
            alt_hypothesis="Difference exists",
        )
        assert result.is_significant
        assert result.sample_sizes["pairs"] == 200


# ===================================================================
# Regression & VIF
# ===================================================================


class TestComputeVIF:
    """Tests for compute_vif function."""

    def test_independent_features(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            {
                "x1": rng.normal(0, 1, 200),
                "x2": rng.normal(0, 1, 200),
                "x3": rng.normal(0, 1, 200),
            }
        )
        vif_df = compute_vif(X)
        assert len(vif_df) == 3
        assert all(vif_df["VIF"] < 5)  # independent → low VIF

    def test_collinear_features(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        X = pd.DataFrame(
            {
                "x": x,
                "x_copy": x + rng.normal(0, 0.01, 200),  # near-perfect copy
                "independent": rng.normal(0, 1, 200),
            }
        )
        vif_df = compute_vif(X)
        # x and x_copy should have very high VIF
        max_vif = vif_df["VIF"].max()
        assert max_vif > 50


class TestOLSRegression:
    """Tests for ols_regression function."""

    def test_basic_regression(self):
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = 10 + 3 * x1 - 2 * x2 + rng.normal(0, 1, n)

        X = pd.DataFrame({"x1": x1, "x2": x2})
        result = ols_regression(X, pd.Series(y), log_transform_y=False)

        assert result.r_squared > 0.7
        assert result.n_observations == n
        assert len(result.coefficients) == 3  # const, x1, x2
        assert len(result.vif_scores) == 2  # x1, x2


# ===================================================================
# Multiple Comparison Corrections
# ===================================================================


class TestApplyCorrection:
    """Tests for apply_correction function."""

    def test_bonferroni(self):
        p_values = [0.01, 0.04, 0.03, 0.005]
        result = apply_correction(p_values, method="bonferroni")
        assert len(result) == 4
        # Adjusted p-values should be ≥ raw
        assert all(result["p-value (bonferroni)"].iloc[i] >= p_values[i] for i in range(4))

    def test_holm(self):
        p_values = [0.01, 0.04, 0.03, 0.005]
        result = apply_correction(p_values, method="holm")
        assert len(result) == 4

    def test_fdr_bh(self):
        p_values = [0.01, 0.04, 0.03, 0.005]
        result = apply_correction(p_values, method="fdr_bh")
        assert len(result) == 4

    def test_with_labels(self):
        result = apply_correction(
            [0.01, 0.03],
            labels=["H1", "H2"],
            method="bonferroni",
        )
        assert list(result["Test"]) == ["H1", "H2"]


# ===================================================================
# Display Helpers
# ===================================================================


class TestFormatTestResult:
    """Tests for format_test_result function."""

    def test_produces_markdown(self, two_distinct_groups):
        a, b = two_distinct_groups
        result = two_group_test(
            a,
            b,
            hypothesis_id="H_fmt",
            null_hypothesis="No difference",
            alt_hypothesis="Difference exists",
        )
        md = format_test_result(result)
        assert "H_fmt" in md
        assert "p-value" in md
        assert "Effect Size" in md


class TestFormatCITable:
    """Tests for format_ci_table function."""

    def test_produces_dataframe(self, normal_data: np.ndarray):
        intervals = {
            "Group A": bootstrap_ci(normal_data),
            "Group B": bootstrap_ci(normal_data + 10),
        }
        table = format_ci_table(intervals)
        assert len(table) == 2
        assert "Mean" in table.columns
