"""Reusable statistical analysis utilities for Airbnb Market Intelligence.

Provides assumption checking, hypothesis testing wrappers, effect size
calculations, confidence intervals, and multicollinearity diagnostics.

All functions operate on numpy arrays or pandas Series — they have no
dependency on DuckDB, Polars, or the pipeline modules. This ensures
they are testable in isolation and reusable across notebooks.

Usage:
    from notebooks.statistics.stats_utils import (
        two_group_test, multi_group_test, bootstrap_ci,
        cohens_d, compute_vif, ols_regression, apply_correction,
    )
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ===================================================================
# Data Classes
# ===================================================================


@dataclass(frozen=True)
class AssumptionResult:
    """Outcome of a distributional assumption check."""

    test_name: str
    statistic: float
    p_value: float
    is_satisfied: bool  # p > alpha → assumption holds
    sample_size: int
    note: str


@dataclass(frozen=True)
class ConfidenceInterval:
    """Bootstrap or analytical confidence interval."""

    mean: float
    ci_lower: float
    ci_upper: float
    ci_level: float  # e.g. 0.95
    method: str  # "bootstrap" or "analytical"
    n: int


@dataclass
class HypothesisTestResult:
    """Complete result of a hypothesis test with effect size."""

    test_name: str
    hypothesis_id: str
    null_hypothesis: str
    alt_hypothesis: str
    test_statistic: float
    p_value: float
    effect_size: float
    effect_size_label: str  # "Cohen's d", "rank-biserial r", etc.
    effect_magnitude: str  # "negligible"/"small"/"medium"/"large"
    is_significant: bool
    alpha: float
    sample_sizes: dict[str, int]
    assumptions_checked: list[AssumptionResult] = field(default_factory=list)
    test_selection_rationale: str = ""
    conclusion: str = ""
    posthoc_results: pd.DataFrame | None = None


@dataclass(frozen=True)
class RegressionResult:
    """OLS regression summary with diagnostics."""

    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_p_value: float
    coefficients: pd.DataFrame  # Feature, coef, std_err, t, p, ci_lower, ci_upper
    vif_scores: pd.DataFrame  # Feature, VIF
    n_observations: int
    residual_normality_p: float  # Jarque-Bera p-value
    heteroscedasticity_p: float  # Breusch-Pagan p-value
    log_transformed: bool
    warnings: list[str] = field(default_factory=list)


# ===================================================================
# Assumption Checks
# ===================================================================


def check_normality(
    data: np.ndarray,
    alpha: float = 0.05,
    max_sample_for_shapiro: int = 5000,
) -> AssumptionResult:
    """Test for normality using Shapiro-Wilk (n ≤ 5000) or D'Agostino-Pearson.

    For large samples (n > 5000), Shapiro-Wilk becomes computationally
    expensive and overly sensitive (rejects even minor deviations).
    D'Agostino-Pearson is more practical at scale.

    Args:
        data: 1-D array of values to test.
        alpha: Significance level.
        max_sample_for_shapiro: Cutoff for switching to D'Agostino.

    Returns:
        AssumptionResult with test outcome and interpretation.
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]
    n = len(clean)

    if n < 8:
        return AssumptionResult(
            test_name="Insufficient data",
            statistic=np.nan,
            p_value=np.nan,
            is_satisfied=False,
            sample_size=n,
            note=f"Only {n} observations — cannot assess normality.",
        )

    if n <= max_sample_for_shapiro:
        stat, p = sp_stats.shapiro(clean)
        test_name = "Shapiro-Wilk"
    else:
        stat, p = sp_stats.normaltest(clean)
        test_name = "D'Agostino-Pearson"

    is_normal = p > alpha
    note = (
        f"{test_name} statistic={stat:.4f}, p={p:.4e}. "
        f"{'Cannot reject normality' if is_normal else 'Normality rejected'} "
        f"at α={alpha} (n={n:,})."
    )

    return AssumptionResult(
        test_name=test_name,
        statistic=stat,
        p_value=p,
        is_satisfied=is_normal,
        sample_size=n,
        note=note,
    )


def check_variance_homogeneity(
    *groups: np.ndarray,
    alpha: float = 0.05,
) -> AssumptionResult:
    """Test equal variance using Levene's test (median-based variant).

    Uses the median variant because it is robust to non-normality,
    which is expected in Airbnb price distributions.

    Args:
        *groups: Two or more arrays of observations.
        alpha: Significance level.

    Returns:
        AssumptionResult with test outcome.
    """
    cleaned = [np.asarray(g, dtype=float)[~np.isnan(np.asarray(g, dtype=float))] for g in groups]
    sizes = [len(g) for g in cleaned]

    if any(s < 2 for s in sizes):
        return AssumptionResult(
            test_name="Levene (median)",
            statistic=np.nan,
            p_value=np.nan,
            is_satisfied=False,
            sample_size=sum(sizes),
            note="One or more groups have fewer than 2 observations.",
        )

    stat, p = sp_stats.levene(*cleaned, center="median")
    is_equal = p > alpha
    note = (
        f"Levene (median) statistic={stat:.4f}, p={p:.4e}. "
        f"{'Equal variance assumption holds' if is_equal else 'Unequal variances detected'} "
        f"at α={alpha}."
    )

    return AssumptionResult(
        test_name="Levene (median)",
        statistic=stat,
        p_value=p,
        is_satisfied=is_equal,
        sample_size=sum(sizes),
        note=note,
    )


def check_independence_note(context: str) -> AssumptionResult:
    """Document the independence assumption (cannot be tested statistically).

    Independence is guaranteed by study design (e.g., one row per listing)
    or violated by design (e.g., repeated calendar observations). This
    function creates a documented record of the reasoning.

    Args:
        context: Plain-English explanation of why independence holds/fails.

    Returns:
        AssumptionResult documenting the assessment.
    """
    return AssumptionResult(
        test_name="Independence (by design)",
        statistic=np.nan,
        p_value=np.nan,
        is_satisfied=True,
        sample_size=0,
        note=context,
    )


# ===================================================================
# Effect Sizes
# ===================================================================


def _magnitude_label(value: float, thresholds: dict[str, float]) -> str:
    """Classify an effect size into negligible/small/medium/large."""
    abs_val = abs(value)
    for label in ["large", "medium", "small"]:
        if abs_val >= thresholds[label]:
            return label
    return "negligible"


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> tuple[float, str]:
    """Compute Cohen's d for two independent groups using pooled SD.

    Interpretation thresholds (Cohen, 1988):
      - |d| < 0.2: negligible
      - 0.2 ≤ |d| < 0.5: small
      - 0.5 ≤ |d| < 0.8: medium
      - |d| ≥ 0.8: large

    Returns:
        (d, magnitude_label) tuple.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]

    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0, "negligible"

    pooled_sd = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
        / (na + nb - 2)
    )

    if pooled_sd == 0:
        return 0.0, "negligible"

    d = (np.mean(a) - np.mean(b)) / pooled_sd
    magnitude = _magnitude_label(d, {"small": 0.2, "medium": 0.5, "large": 0.8})
    return round(d, 4), magnitude


def rank_biserial_correlation(
    u_statistic: float, n1: int, n2: int
) -> tuple[float, str]:
    """Effect size for Mann-Whitney U test.

    r = 1 - (2U / n1·n2). Range: [-1, 1].

    Interpretation (Kerby, 2014):
      - |r| < 0.1: negligible
      - 0.1 ≤ |r| < 0.3: small
      - 0.3 ≤ |r| < 0.5: medium
      - |r| ≥ 0.5: large

    Returns:
        (r, magnitude_label) tuple.
    """
    if n1 * n2 == 0:
        return 0.0, "negligible"

    r = 1 - (2 * u_statistic) / (n1 * n2)
    magnitude = _magnitude_label(r, {"small": 0.1, "medium": 0.3, "large": 0.5})
    return round(r, 4), magnitude


def eta_squared(
    f_statistic: float, df_between: int, df_within: int
) -> tuple[float, str]:
    """Compute eta-squared from ANOVA / Kruskal-Wallis F or H statistic.

    η² ≈ (F × df_between) / (F × df_between + df_within)

    Interpretation (Cohen, 1988):
      - η² < 0.01: negligible
      - 0.01 ≤ η² < 0.06: small
      - 0.06 ≤ η² < 0.14: medium
      - η² ≥ 0.14: large

    Returns:
        (eta_sq, magnitude_label) tuple.
    """
    denominator = f_statistic * df_between + df_within
    if denominator == 0:
        return 0.0, "negligible"

    eta_sq = (f_statistic * df_between) / denominator
    magnitude = _magnitude_label(eta_sq, {"small": 0.01, "medium": 0.06, "large": 0.14})
    return round(eta_sq, 4), magnitude


def epsilon_squared(h_statistic: float, n: int, k: int) -> tuple[float, str]:
    """Effect size for Kruskal-Wallis H-test.

    ε² = (H - k + 1) / (n - k)

    Uses same interpretation thresholds as eta-squared.

    Returns:
        (epsilon_sq, magnitude_label) tuple.
    """
    denominator = n - k
    if denominator <= 0:
        return 0.0, "negligible"

    eps_sq = (h_statistic - k + 1) / denominator
    eps_sq = max(0.0, eps_sq)  # can be negative in edge cases
    magnitude = _magnitude_label(eps_sq, {"small": 0.01, "medium": 0.06, "large": 0.14})
    return round(eps_sq, 4), magnitude


# ===================================================================
# Confidence Intervals
# ===================================================================


def bootstrap_ci(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> ConfidenceInterval:
    """Non-parametric bootstrap confidence interval.

    Preferred over analytical CIs when the underlying distribution
    is non-normal (right-skewed price data).

    Args:
        data: 1-D array of observations.
        statistic_func: Function to compute the statistic (default: mean).
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (0.95 = 95% CI).
        random_state: Seed for reproducibility.

    Returns:
        ConfidenceInterval with point estimate and bounds.
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]
    n = len(clean)

    if n == 0:
        return ConfidenceInterval(
            mean=np.nan, ci_lower=np.nan, ci_upper=np.nan,
            ci_level=ci_level, method="bootstrap", n=0,
        )

    rng = np.random.default_rng(random_state)
    boot_stats = np.array([
        statistic_func(rng.choice(clean, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - ci_level
    lower = np.percentile(boot_stats, alpha / 2 * 100)
    upper = np.percentile(boot_stats, (1 - alpha / 2) * 100)

    return ConfidenceInterval(
        mean=float(statistic_func(clean)),
        ci_lower=round(float(lower), 4),
        ci_upper=round(float(upper), 4),
        ci_level=ci_level,
        method="bootstrap",
        n=n,
    )


def analytical_ci(
    data: np.ndarray,
    ci_level: float = 0.95,
) -> ConfidenceInterval:
    """Analytical (t-distribution) confidence interval for the mean.

    Valid when n is large (CLT applies) or data is approximately normal.
    For skewed distributions, prefer ``bootstrap_ci``.

    Args:
        data: 1-D array of observations.
        ci_level: Confidence level.

    Returns:
        ConfidenceInterval with point estimate and bounds.
    """
    clean = np.asarray(data, dtype=float)
    clean = clean[~np.isnan(clean)]
    n = len(clean)

    if n < 2:
        return ConfidenceInterval(
            mean=np.nan, ci_lower=np.nan, ci_upper=np.nan,
            ci_level=ci_level, method="analytical", n=n,
        )

    mean = float(np.mean(clean))
    se = float(sp_stats.sem(clean))
    alpha = 1 - ci_level
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df=n - 1)

    return ConfidenceInterval(
        mean=mean,
        ci_lower=round(mean - t_crit * se, 4),
        ci_upper=round(mean + t_crit * se, 4),
        ci_level=ci_level,
        method="analytical",
        n=n,
    )


# ===================================================================
# Hypothesis Test Wrappers
# ===================================================================


def two_group_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    hypothesis_id: str,
    null_hypothesis: str,
    alt_hypothesis: str,
    group_a_label: str = "Group A",
    group_b_label: str = "Group B",
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> HypothesisTestResult:
    """Automated two-group comparison with assumption-driven test selection.

    Decision tree:
      1. Check normality of both groups.
      2. If both normal → check equal variance → Welch's t-test.
      3. If either non-normal → Mann-Whitney U (non-parametric).

    Args:
        group_a: Observations for the first group.
        group_b: Observations for the second group.
        hypothesis_id: Label (e.g., "H1").
        null_hypothesis: Plain-English H₀.
        alt_hypothesis: Plain-English H₁.
        group_a_label: Human label for group A.
        group_b_label: Human label for group B.
        alpha: Significance level.
        alternative: "two-sided", "greater", or "less".

    Returns:
        HypothesisTestResult with full methodology documentation.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]

    assumptions: list[AssumptionResult] = []

    # Independence (by design — one observation per listing)
    assumptions.append(check_independence_note(
        "Each observation represents a unique listing. Independence holds by design."
    ))

    # Normality
    norm_a = check_normality(a, alpha=alpha)
    norm_b = check_normality(b, alpha=alpha)
    assumptions.extend([norm_a, norm_b])

    use_parametric = norm_a.is_satisfied and norm_b.is_satisfied

    if use_parametric:
        # Check equal variance → Welch's t-test (doesn't assume equal var)
        var_check = check_variance_homogeneity(a, b, alpha=alpha)
        assumptions.append(var_check)

        stat, p = sp_stats.ttest_ind(a, b, equal_var=False, alternative=alternative)
        test_name = "Welch's t-test"
        d, d_mag = cohens_d(a, b)
        effect_label = "Cohen's d"
        rationale = (
            "Both groups passed normality tests. Welch's t-test was selected "
            "as the parametric test because it does not assume equal variances."
        )
    else:
        stat, p = sp_stats.mannwhitneyu(a, b, alternative=alternative)
        test_name = "Mann-Whitney U"
        r, d_mag = rank_biserial_correlation(stat, len(a), len(b))
        d = r
        effect_label = "rank-biserial r"
        # Also compute Cohen's d on raw values for interpretability
        d_raw, _ = cohens_d(a, b)
        rationale = (
            f"At least one group failed the normality test "
            f"(p < {alpha}). Mann-Whitney U was selected as the non-parametric "
            f"alternative. Supplementary Cohen's d on raw values = {d_raw:.3f}."
        )

    is_sig = p < alpha
    conclusion = _build_conclusion(
        test_name, hypothesis_id, is_sig, p, d, effect_label,
        d_mag, group_a_label, group_b_label, a, b,
    )

    return HypothesisTestResult(
        test_name=test_name,
        hypothesis_id=hypothesis_id,
        null_hypothesis=null_hypothesis,
        alt_hypothesis=alt_hypothesis,
        test_statistic=round(float(stat), 4),
        p_value=float(p),
        effect_size=d,
        effect_size_label=effect_label,
        effect_magnitude=d_mag,
        is_significant=is_sig,
        alpha=alpha,
        sample_sizes={group_a_label: len(a), group_b_label: len(b)},
        assumptions_checked=assumptions,
        test_selection_rationale=rationale,
        conclusion=conclusion,
    )


def multi_group_test(
    groups: dict[str, np.ndarray],
    hypothesis_id: str,
    null_hypothesis: str,
    alt_hypothesis: str,
    alpha: float = 0.05,
    max_posthoc_groups: int = 20,
) -> HypothesisTestResult:
    """Multi-group comparison (ANOVA or Kruskal-Wallis) with post-hoc.

    Decision tree:
      1. Check normality of all groups.
      2. If all normal → one-way ANOVA.
      3. If any non-normal → Kruskal-Wallis H.

    Post-hoc: If significant, runs Dunn's test via pairwise
    Mann-Whitney U with Bonferroni correction.

    Args:
        groups: {group_name: observations_array}.
        hypothesis_id: Label (e.g., "H4").
        null_hypothesis: Plain-English H₀.
        alt_hypothesis: Plain-English H₁.
        alpha: Significance level.
        max_posthoc_groups: Skip post-hoc if more groups than this.

    Returns:
        HypothesisTestResult with posthoc_results DataFrame if significant.
    """
    cleaned = {
        name: np.asarray(arr, dtype=float)[~np.isnan(np.asarray(arr, dtype=float))]
        for name, arr in groups.items()
        if len(np.asarray(arr, dtype=float)[~np.isnan(np.asarray(arr, dtype=float))]) >= 2
    }

    group_names = list(cleaned.keys())
    group_arrays = list(cleaned.values())
    k = len(group_names)
    n_total = sum(len(g) for g in group_arrays)

    assumptions: list[AssumptionResult] = []
    assumptions.append(check_independence_note(
        "Each observation represents a unique listing. Independence holds by design."
    ))

    # Check normality of all groups (sample 3 if many groups)
    norm_checks = []
    check_subset = group_names[:5] if k > 5 else group_names
    for name in check_subset:
        norm = check_normality(cleaned[name], alpha=alpha)
        norm_checks.append(norm)
        assumptions.append(norm)

    all_normal = all(nc.is_satisfied for nc in norm_checks)

    if all_normal:
        stat, p = sp_stats.f_oneway(*group_arrays)
        test_name = "One-way ANOVA"
        df_between = k - 1
        df_within = n_total - k
        es, es_mag = eta_squared(stat, df_between, df_within)
        effect_label = "η² (eta-squared)"
        rationale = (
            f"All sampled groups passed normality tests. One-way ANOVA was "
            f"selected. k={k} groups, total n={n_total:,}."
        )
    else:
        stat, p = sp_stats.kruskal(*group_arrays)
        test_name = "Kruskal-Wallis H"
        es, es_mag = epsilon_squared(stat, n_total, k)
        effect_label = "ε² (epsilon-squared)"
        rationale = (
            f"At least one group failed normality. Kruskal-Wallis H was "
            f"selected as the non-parametric alternative. k={k}, n={n_total:,}."
        )

    is_sig = p < alpha

    # Post-hoc pairwise comparisons (if significant and tractable)
    posthoc_df = None
    if is_sig and k <= max_posthoc_groups:
        posthoc_df = _pairwise_posthoc(cleaned, alpha)

    conclusion = (
        f"[{hypothesis_id}] {test_name}: H={stat:.2f}, p={p:.2e}. "
        f"Effect size {effect_label} = {es:.4f} ({es_mag}). "
        f"{'Reject H₀' if is_sig else 'Fail to reject H₀'} at α={alpha}."
    )

    return HypothesisTestResult(
        test_name=test_name,
        hypothesis_id=hypothesis_id,
        null_hypothesis=null_hypothesis,
        alt_hypothesis=alt_hypothesis,
        test_statistic=round(float(stat), 4),
        p_value=float(p),
        effect_size=es,
        effect_size_label=effect_label,
        effect_magnitude=es_mag,
        is_significant=is_sig,
        alpha=alpha,
        sample_sizes={name: len(arr) for name, arr in cleaned.items()},
        assumptions_checked=assumptions,
        test_selection_rationale=rationale,
        conclusion=conclusion,
        posthoc_results=posthoc_df,
    )


def paired_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    hypothesis_id: str,
    null_hypothesis: str,
    alt_hypothesis: str,
    label_a: str = "Condition A",
    label_b: str = "Condition B",
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> HypothesisTestResult:
    """Paired comparison using Wilcoxon signed-rank or paired t-test.

    Designed for H5 (weekend vs weekday) where observations are paired
    at the listing level.

    Decision tree:
      1. Compute paired differences.
      2. Check normality of differences.
      3. If normal → paired t-test.
      4. If non-normal → Wilcoxon signed-rank test.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)

    # Align and remove pairs with NaN in either
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diffs = a - b
    n = len(diffs)

    assumptions: list[AssumptionResult] = []
    assumptions.append(check_independence_note(
        "Observations are paired at the listing level. Each listing contributes "
        "one paired difference (mean weekend price - mean weekday price)."
    ))

    norm_diff = check_normality(diffs, alpha=alpha)
    assumptions.append(norm_diff)

    if norm_diff.is_satisfied:
        stat, p = sp_stats.ttest_rel(a, b, alternative=alternative)
        test_name = "Paired t-test"
        d, d_mag = cohens_d(a, b)
        effect_label = "Cohen's d (paired)"
        rationale = "Paired differences are approximately normal. Paired t-test selected."
    else:
        # Remove zero differences for Wilcoxon
        nonzero = diffs[diffs != 0]
        if len(nonzero) < 10:
            stat, p = np.nan, 1.0
            test_name = "Wilcoxon (insufficient non-zero diffs)"
            d, d_mag = 0.0, "negligible"
        else:
            stat, p = sp_stats.wilcoxon(nonzero, alternative=alternative)
            test_name = "Wilcoxon signed-rank"
            d, d_mag = cohens_d(a, b)
        effect_label = "Cohen's d (paired)"
        rationale = (
            "Paired differences are non-normal. Wilcoxon signed-rank test "
            "selected as the non-parametric paired alternative."
        )

    is_sig = p < alpha
    mean_diff = float(np.mean(diffs))
    conclusion = (
        f"[{hypothesis_id}] {test_name}: stat={stat:.2f}, p={p:.2e}. "
        f"Mean paired difference = {mean_diff:.2f}. "
        f"Effect size {effect_label} = {d:.4f} ({d_mag}). "
        f"{'Reject H₀' if is_sig else 'Fail to reject H₀'} at α={alpha}."
    )

    return HypothesisTestResult(
        test_name=test_name,
        hypothesis_id=hypothesis_id,
        null_hypothesis=null_hypothesis,
        alt_hypothesis=alt_hypothesis,
        test_statistic=round(float(stat), 4) if not np.isnan(stat) else 0.0,
        p_value=float(p),
        effect_size=d,
        effect_size_label=effect_label,
        effect_magnitude=d_mag,
        is_significant=is_sig,
        alpha=alpha,
        sample_sizes={label_a: n, label_b: n, "pairs": n},
        assumptions_checked=assumptions,
        test_selection_rationale=rationale,
        conclusion=conclusion,
    )


# ===================================================================
# Post-hoc Pairwise Comparisons
# ===================================================================


def _pairwise_posthoc(
    groups: dict[str, np.ndarray],
    alpha: float,
) -> pd.DataFrame:
    """Run pairwise Mann-Whitney U with Bonferroni correction."""
    from itertools import combinations

    names = list(groups.keys())
    records = []
    pairs = list(combinations(names, 2))
    m = len(pairs)  # number of comparisons

    for name_a, name_b in pairs:
        u, p = sp_stats.mannwhitneyu(groups[name_a], groups[name_b], alternative="two-sided")
        p_adj = min(p * m, 1.0)  # Bonferroni
        r, mag = rank_biserial_correlation(u, len(groups[name_a]), len(groups[name_b]))
        records.append({
            "Group A": name_a,
            "Group B": name_b,
            "U statistic": round(u, 1),
            "p-value (raw)": p,
            "p-value (Bonferroni)": round(p_adj, 6),
            "Significant": p_adj < alpha,
            "Effect size (r)": r,
            "Magnitude": mag,
        })

    return pd.DataFrame(records).sort_values("p-value (raw)")


# ===================================================================
# Regression & Multicollinearity
# ===================================================================


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each feature.

    VIF > 5: moderate multicollinearity.
    VIF > 10: severe multicollinearity — consider removing.

    Args:
        X: Feature matrix (numeric columns only, no intercept).

    Returns:
        DataFrame with columns [Feature, VIF, Flag].
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Ensure no constant columns
    X_clean = X.select_dtypes(include=[np.number]).dropna()
    X_values = X_clean.values

    records = []
    for i, col in enumerate(X_clean.columns):
        try:
            vif_val = variance_inflation_factor(X_values, i)
        except (np.linalg.LinAlgError, ZeroDivisionError):
            vif_val = np.inf

        flag = ""
        if vif_val > 10:
            flag = "⚠️ SEVERE"
        elif vif_val > 5:
            flag = "⚡ MODERATE"

        records.append({"Feature": col, "VIF": round(vif_val, 2), "Flag": flag})

    return pd.DataFrame(records).sort_values("VIF", ascending=False)


def ols_regression(
    X: pd.DataFrame,
    y: pd.Series,
    log_transform_y: bool = True,
) -> RegressionResult:
    """Run OLS regression with comprehensive diagnostics.

    Steps:
      1. Optionally log-transform y (common for price data).
      2. Add constant and fit OLS via statsmodels.
      3. Compute VIF for all predictors.
      4. Test residual normality (Jarque-Bera).
      5. Test heteroscedasticity (Breusch-Pagan).

    Args:
        X: Feature matrix (numeric, may include dummies).
        y: Target variable (typically price_local).
        log_transform_y: If True, use log1p(y) as target.

    Returns:
        RegressionResult with full diagnostic suite.
    """
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan

    warnings_list: list[str] = []

    # Align and clean
    combined = pd.concat([X, y.rename("_target")], axis=1).dropna()
    X_clean = combined.drop(columns=["_target"])
    y_clean = combined["_target"]

    if log_transform_y:
        y_clean = np.log1p(y_clean)

    # Drop zero-variance columns
    zero_var = X_clean.columns[X_clean.std() == 0]
    if len(zero_var) > 0:
        warnings_list.append(f"Dropped zero-variance columns: {list(zero_var)}")
        X_clean = X_clean.drop(columns=zero_var)

    X_with_const = sm.add_constant(X_clean, has_constant="add")

    # Fit OLS
    model = sm.OLS(y_clean, X_with_const).fit()

    # VIF (exclude constant)
    vif_df = compute_vif(X_clean)

    # Coefficient table
    coef_records = []
    for name in model.params.index:
        coef_records.append({
            "Feature": name,
            "Coefficient": round(model.params[name], 6),
            "Std Error": round(model.bse[name], 6),
            "t-statistic": round(model.tvalues[name], 3),
            "p-value": model.pvalues[name],
            "CI Lower": round(model.conf_int().loc[name, 0], 6),
            "CI Upper": round(model.conf_int().loc[name, 1], 6),
        })
    coef_df = pd.DataFrame(coef_records)

    # Residual diagnostics
    residuals = model.resid
    jb_stat, jb_p, _, _ = sp_stats.jarque_bera(residuals)

    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_with_const)
    except Exception:
        bp_p = np.nan
        warnings_list.append("Breusch-Pagan test failed — check for perfect collinearity.")

    if jb_p < 0.05:
        warnings_list.append("Residuals are non-normal (Jarque-Bera p < 0.05).")
    if not np.isnan(bp_p) and bp_p < 0.05:
        warnings_list.append("Heteroscedasticity detected (Breusch-Pagan p < 0.05).")

    return RegressionResult(
        r_squared=round(model.rsquared, 4),
        adj_r_squared=round(model.rsquared_adj, 4),
        f_statistic=round(model.fvalue, 2),
        f_p_value=model.f_pvalue,
        coefficients=coef_df,
        vif_scores=vif_df,
        n_observations=int(model.nobs),
        residual_normality_p=round(jb_p, 6),
        heteroscedasticity_p=round(bp_p, 6) if not np.isnan(bp_p) else np.nan,
        log_transformed=log_transform_y,
        warnings=warnings_list,
    )


# ===================================================================
# Multiple Comparison Corrections
# ===================================================================


def apply_correction(
    p_values: list[float],
    labels: list[str] | None = None,
    method: str = "holm",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply multiple comparison correction to a set of p-values.

    Methods:
      - bonferroni: p_adj = p × m. Conservative, controls FWER.
      - holm: Step-down Bonferroni. Less conservative, still FWER.
      - fdr_bh: Benjamini-Hochberg. Controls FDR, best for exploratory.

    Args:
        p_values: Raw p-values from repeated tests.
        labels: Optional labels for each test.
        method: Correction method.
        alpha: Family-wise significance level.

    Returns:
        DataFrame with columns [Test, p_raw, p_adjusted, Significant].
    """
    from statsmodels.stats.multitest import multipletests

    m = len(p_values)
    if labels is None:
        labels = [f"Test {i+1}" for i in range(m)]

    reject, p_adj, _, _ = multipletests(p_values, alpha=alpha, method=method)

    return pd.DataFrame({
        "Test": labels,
        "p-value (raw)": [round(p, 6) for p in p_values],
        f"p-value ({method})": [round(p, 6) for p in p_adj],
        "Significant": reject,
    })


# ===================================================================
# Display Helpers
# ===================================================================


def format_test_result(result: HypothesisTestResult) -> str:
    """Format a HypothesisTestResult as publication-ready markdown."""
    sig_icon = "✅" if result.is_significant else "❌"

    lines = [
        f"### {result.hypothesis_id}: {result.test_name} {sig_icon}",
        "",
        f"**H₀:** {result.null_hypothesis}",
        f"**H₁:** {result.alt_hypothesis}",
        "",
        "| Metric | Value |",
        "|:-------|:------|",
        f"| Test Statistic | {result.test_statistic:.4f} |",
        f"| p-value | {result.p_value:.2e} |",
        f"| Effect Size ({result.effect_size_label}) | {result.effect_size:.4f} |",
        f"| Effect Magnitude | {result.effect_magnitude} |",
        f"| Significant at α={result.alpha} | {'Yes' if result.is_significant else 'No'} |",
    ]

    # Sample sizes
    sizes = " | ".join(f"{k}: {v:,}" for k, v in result.sample_sizes.items())
    lines.append(f"| Sample Sizes | {sizes} |")
    lines.append("")

    # Assumptions
    if result.assumptions_checked:
        lines.append("**Assumptions Checked:**")
        for a in result.assumptions_checked:
            icon = "✅" if a.is_satisfied else "⚠️"
            lines.append(f"- {icon} {a.test_name}: {a.note}")
        lines.append("")

    # Rationale
    if result.test_selection_rationale:
        lines.append(f"**Test Selection:** {result.test_selection_rationale}")
        lines.append("")

    # Conclusion
    if result.conclusion:
        lines.append(f"**Conclusion:** {result.conclusion}")

    return "\n".join(lines)


def format_ci_table(intervals: dict[str, ConfidenceInterval]) -> pd.DataFrame:
    """Format multiple CIs into a clean comparison table."""
    records = []
    for label, ci in intervals.items():
        records.append({
            "Group": label,
            "Mean": round(ci.mean, 2),
            f"CI Lower ({ci.ci_level*100:.0f}%)": ci.ci_lower,
            f"CI Upper ({ci.ci_level*100:.0f}%)": ci.ci_upper,
            "Width": round(ci.ci_upper - ci.ci_lower, 2),
            "N": ci.n,
            "Method": ci.method,
        })
    return pd.DataFrame(records)


# ===================================================================
# Internal Helpers
# ===================================================================


def _build_conclusion(
    test_name: str,
    hypothesis_id: str,
    is_sig: bool,
    p: float,
    effect_size: float,
    effect_label: str,
    magnitude: str,
    label_a: str,
    label_b: str,
    a: np.ndarray,
    b: np.ndarray,
) -> str:
    """Build a human-readable conclusion for a two-group test."""
    direction = "higher" if np.median(a) > np.median(b) else "lower"
    diff = abs(np.median(a) - np.median(b))

    if is_sig:
        return (
            f"[{hypothesis_id}] {test_name}: {label_a} has a statistically "
            f"significantly {direction} median than {label_b} "
            f"(p={p:.2e}, {effect_label}={effect_size:.3f}, {magnitude} effect). "
            f"Median difference: {diff:.2f}."
        )
    return (
        f"[{hypothesis_id}] {test_name}: No statistically significant difference "
        f"between {label_a} and {label_b} (p={p:.2e}). "
        f"Effect size {effect_label}={effect_size:.3f} ({magnitude})."
    )
