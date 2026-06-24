"""Model generalisation and bias analysis for §6.4.

Implements three complementary bias analysis strategies:
  1. Cross-neighbourhood generalisation (Leave-One-Neighbourhood-Group-Out)
  2. Cross-city transfer learning (Train on A → Evaluate on B)
  3. Group-level bias detection across configurable dimensions

Produces a comprehensive BiasAuditReport with actionable mitigations.

Usage (from CLI via orchestrator):
    from src.platform.data_science.validation.bias_auditor import run_bias_audit
    report = run_bias_audit(experiment_result, feature_set, split, config)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.platform.common.utils import OUTPUT_DIR

logger = logging.getLogger(__name__)

OUTPUTS_DIR = OUTPUT_DIR / "ml"


# ===================================================================
# Data Classes
# ===================================================================


@dataclass
class LONGOResult:
    """Leave-One-Neighbourhood-Group-Out result for a single group."""

    city: str
    neighbourhood_group: str
    n_train: int
    n_test: int
    mae: float
    mape: float
    mean_residual: float
    gap_vs_cv: float  # LONGO MAE - standard CV MAE


@dataclass
class TransferResult:
    """Cross-city transfer evaluation for a single pair."""

    train_city: str
    eval_city: str
    n_train: int
    n_eval: int
    mae: float
    mape: float
    r2: float
    mae_degradation: float  # transfer MAE - in-city MAE


@dataclass
class GroupBias:
    """Bias metrics for a single group in a stratification dimension."""

    dimension: str
    group: str
    n: int
    mean_residual: float
    ci_lower: float
    ci_upper: float
    is_significant: bool
    cohens_d: float
    cohens_d_magnitude: str
    direction: str  # "over-predicted" or "under-predicted"


@dataclass
class FairnessSummary:
    """Executive summary of the bias audit."""

    overall_risk: str  # "LOW", "MEDIUM", "HIGH"
    total_biases_found: int
    significant_biases: int
    top_biases: list[GroupBias]
    recommendations: list[str]


@dataclass
class BiasAuditReport:
    """Complete bias audit report."""

    experiment_id: str
    model_name: str
    longo_results: list[LONGOResult]
    transfer_matrix: list[TransferResult]
    group_biases: list[GroupBias]
    fairness_summary: FairnessSummary
    feature_ablation: dict[str, float] | None


# ===================================================================
# Metrics Helper
# ===================================================================


def _compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    log_transformed: bool = True,
) -> dict[str, float]:
    """Compute MAE, MAPE, R² on original scale."""
    from sklearn.metrics import mean_absolute_error, r2_score

    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if log_transformed:
        y_t = np.expm1(y_t)
        y_p = np.expm1(y_p)

    y_p = np.maximum(y_p, 0)

    nonzero = y_t > 1.0
    mape = (
        float(np.mean(np.abs(y_t[nonzero] - y_p[nonzero]) / y_t[nonzero]) * 100)
        if nonzero.sum() > 0
        else np.nan
    )

    return {
        "mae": float(mean_absolute_error(y_t, y_p)),
        "mape": mape,
        "r2": float(r2_score(y_t, y_p)) if len(y_t) > 1 else np.nan,
    }


# ===================================================================
# Cross-Neighbourhood Generalisation (LONGO)
# ===================================================================


def cross_neighbourhood_cv(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    config: dict,
    baseline_cv_mae: float = 0.0,
) -> list[LONGOResult]:
    """Leave-One-Neighbourhood-Group-Out cross-validation.

    For each neighbourhood_group in each city, the model is trained on
    all OTHER neighbourhoods and evaluated on the held-out group. This
    tests whether the model can generalise to unseen areas.

    Args:
        model: Scikit-learn compatible estimator (will be cloned).
        X: Full feature matrix.
        y: Full target series.
        metadata: DataFrame with 'city' and 'neighbourhood_group' columns.
        config: ML config dict.
        baseline_cv_mae: Standard CV MAE for gap calculation.

    Returns:
        List of LONGOResult, one per (city, neighbourhood_group).
    """
    longo_cfg = config.get("bias_audit", {}).get("longo", {})
    if not longo_cfg.get("enabled", True):
        logger.info("LONGO CV disabled in config")
        return []

    min_size = longo_cfg.get("min_group_size", 30)
    log_transformed = config["target"].get("transform") == "log1p"
    results: list[LONGOResult] = []

    if "neighbourhood_group" not in metadata.columns or "city" not in metadata.columns:
        logger.warning("Missing neighbourhood_group or city in metadata — skipping LONGO")
        return results

    for city in metadata["city"].unique():
        city_mask = metadata["city"] == city

        for nbhd, group_idx in metadata[city_mask].groupby("neighbourhood_group").groups.items():
            if len(group_idx) < min_size:
                continue

            test_mask = metadata.index.isin(group_idx)
            train_mask = city_mask & ~test_mask

            if train_mask.sum() < min_size:
                continue

            X_train_fold = X.loc[train_mask]
            y_train_fold = y.loc[train_mask]
            X_test_fold = X.loc[test_mask]
            y_test_fold = y.loc[test_mask]

            try:
                fold_model = clone(model)
                fold_model.fit(X_train_fold, y_train_fold)
                y_pred = fold_model.predict(X_test_fold)

                metrics = _compute_regression_metrics(y_test_fold.values, y_pred, log_transformed)

                results.append(
                    LONGOResult(
                        city=str(city),
                        neighbourhood_group=str(nbhd),
                        n_train=int(train_mask.sum()),
                        n_test=int(test_mask.sum()),
                        mae=metrics["mae"],
                        mape=metrics["mape"],
                        mean_residual=(
                            float(np.mean(np.expm1(y_test_fold.values) - np.expm1(y_pred)))
                            if log_transformed
                            else float(np.mean(y_test_fold.values - y_pred))
                        ),
                        gap_vs_cv=metrics["mae"] - baseline_cv_mae,
                    )
                )
            except Exception as exc:
                logger.warning("LONGO failed for %s/%s: %s", city, nbhd, exc)

    logger.info("LONGO CV: evaluated %d neighbourhood groups", len(results))
    return results


# ===================================================================
# Cross-City Transfer Learning
# ===================================================================


def cross_city_transfer(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    config: dict,
) -> list[TransferResult]:
    """Train on City A, evaluate on City B for all city pairs.

    Tests whether pricing relationships learned in one city generalise
    to another. Features containing neighbourhood or city-specific
    information can optionally be excluded for fairer comparison.

    Args:
        model: Scikit-learn compatible estimator (will be cloned).
        X: Full feature matrix.
        y: Full target series.
        metadata: DataFrame with 'city' column.
        config: ML config dict.

    Returns:
        List of TransferResult for all city pair combinations.
    """
    transfer_cfg = config.get("bias_audit", {}).get("cross_city_transfer", {})
    if not transfer_cfg.get("enabled", True):
        logger.info("Cross-city transfer disabled in config")
        return []

    log_transformed = config["target"].get("transform") == "log1p"

    if "city" not in metadata.columns:
        logger.warning("No 'city' column in metadata — skipping cross-city transfer")
        return []

    cities = sorted(metadata["city"].unique())
    if len(cities) < 2:
        logger.warning("Only %d city found — cross-city transfer requires ≥ 2", len(cities))
        return []

    # Optionally exclude city-specific features
    exclude_patterns = transfer_cfg.get("exclude_features_pattern", [])
    transfer_cols = [c for c in X.columns if not any(c.startswith(pat) for pat in exclude_patterns)]
    X_transfer = X[transfer_cols]

    # Compute in-city baselines
    in_city_mae: dict[str, float] = {}
    for city in cities:
        mask = metadata["city"] == city
        if mask.sum() < 50:
            continue
        city_model = clone(model)
        city_X = X_transfer.loc[mask]
        city_y = y.loc[mask]

        # Simple 80/20 split for baseline
        n = len(city_X)
        split_idx = int(n * 0.8)
        city_model.fit(city_X.iloc[:split_idx], city_y.iloc[:split_idx])
        y_pred = city_model.predict(city_X.iloc[split_idx:])
        metrics = _compute_regression_metrics(
            city_y.iloc[split_idx:].values, y_pred, log_transformed
        )
        in_city_mae[city] = metrics["mae"]

    # Cross-city transfer
    results: list[TransferResult] = []

    for train_city in cities:
        train_mask = metadata["city"] == train_city
        if train_mask.sum() < 50:
            continue

        train_model = clone(model)
        train_model.fit(X_transfer.loc[train_mask], y.loc[train_mask])

        for eval_city in cities:
            eval_mask = metadata["city"] == eval_city
            if eval_mask.sum() < 50:
                continue

            y_pred = train_model.predict(X_transfer.loc[eval_mask])
            metrics = _compute_regression_metrics(y.loc[eval_mask].values, y_pred, log_transformed)

            baseline = in_city_mae.get(eval_city, 0.0) if train_city != eval_city else 0.0
            degradation = metrics["mae"] - baseline if train_city != eval_city else 0.0

            results.append(
                TransferResult(
                    train_city=train_city,
                    eval_city=eval_city,
                    n_train=int(train_mask.sum()),
                    n_eval=int(eval_mask.sum()),
                    mae=metrics["mae"],
                    mape=metrics["mape"],
                    r2=metrics["r2"],
                    mae_degradation=degradation,
                )
            )

    logger.info("Cross-city transfer: evaluated %d city pairs", len(results))
    return results


# ===================================================================
# Group-Level Bias Detection
# ===================================================================


def _bootstrap_mean_ci(
    residuals: np.ndarray,
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap CI for the mean residual."""
    rng = np.random.default_rng(random_state)
    boot_means = np.array(
        [
            np.mean(rng.choice(residuals, size=len(residuals), replace=True))
            for _ in range(n_bootstrap)
        ]
    )
    alpha = 1 - ci_level
    lower = float(np.percentile(boot_means, alpha / 2 * 100))
    upper = float(np.percentile(boot_means, (1 - alpha / 2) * 100))
    mean = float(np.mean(residuals))
    return mean, lower, upper


def _cohens_d(group_residuals: np.ndarray, all_residuals: np.ndarray) -> tuple[float, str]:
    """Cohen's d of group residuals vs overall residuals."""
    n_g, n_a = len(group_residuals), len(all_residuals)
    if n_g < 2 or n_a < 2:
        return 0.0, "negligible"

    pooled_sd = np.sqrt(
        ((n_g - 1) * np.var(group_residuals, ddof=1) + (n_a - 1) * np.var(all_residuals, ddof=1))
        / (n_g + n_a - 2)
    )
    if pooled_sd == 0:
        return 0.0, "negligible"

    d = float((np.mean(group_residuals) - np.mean(all_residuals)) / pooled_sd)
    abs_d = abs(d)

    if abs_d >= 0.8:
        return round(d, 4), "large"
    if abs_d >= 0.5:
        return round(d, 4), "medium"
    if abs_d >= 0.2:
        return round(d, 4), "small"
    return round(d, 4), "negligible"


def compute_group_bias(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
    config: dict,
) -> list[GroupBias]:
    """Detect systematic prediction bias across configurable dimensions.

    For each group in each dimension, computes:
      - Mean residual (bias direction)
      - Bootstrap CI (statistical significance)
      - Cohen's d (effect size vs overall residuals)

    Args:
        y_true: True target values.
        y_pred: Predicted values.
        metadata: DataFrame with grouping columns.
        config: ML config dict.

    Returns:
        List of GroupBias results.
    """
    log_transformed = config["target"].get("transform") == "log1p"
    bias_cfg = config.get("bias_audit", {})
    dimensions = bias_cfg.get("dimensions", [])
    alpha = bias_cfg.get("significance_alpha", 0.05)
    n_boot = bias_cfg.get("bootstrap_iterations", 5000)

    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if log_transformed:
        y_t = np.expm1(y_t)
        y_p = np.expm1(y_p)

    y_p = np.maximum(y_p, 0)
    all_residuals = y_t - y_p

    results: list[GroupBias] = []

    for dim_cfg in dimensions:
        col = dim_cfg["column"]
        if col not in metadata.columns:
            logger.warning("Bias dimension '%s' not in metadata", col)
            continue

        for group_val, group_idx in metadata.groupby(col).groups.items():
            n = len(group_idx)
            if n < 10:
                continue

            mask = metadata.index.isin(group_idx)
            group_resid = all_residuals[mask]

            mean_r, ci_lo, ci_hi = _bootstrap_mean_ci(group_resid, n_boot)
            d, d_mag = _cohens_d(group_resid, all_residuals)

            # Significant if CI doesn't contain zero
            significant = not (ci_lo <= 0 <= ci_hi)

            direction = "under-predicted" if mean_r > 0 else "over-predicted"

            results.append(
                GroupBias(
                    dimension=col,
                    group=str(group_val),
                    n=n,
                    mean_residual=round(mean_r, 2),
                    ci_lower=round(ci_lo, 2),
                    ci_upper=round(ci_hi, 2),
                    is_significant=significant,
                    cohens_d=d,
                    cohens_d_magnitude=d_mag,
                    direction=direction,
                )
            )

    logger.info("Group bias analysis: %d groups tested", len(results))
    return results


# ===================================================================
# Feature Ablation
# ===================================================================


def neighbourhood_feature_ablation(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
) -> dict[str, float]:
    """Measure performance impact of removing neighbourhood features.

    Trains the model with and without neighbourhood-related columns
    to quantify how much the model relies on location memorisation
    vs fundamental pricing factors.

    Returns:
        Dict with MAE_with_neighbourhood, MAE_without, and degradation.
    """
    from sklearn.model_selection import cross_val_score

    log_transformed = config["target"].get("transform") == "log1p"

    # Identify neighbourhood columns
    nbhd_cols = [c for c in X.columns if "neighbourhood" in c.lower() or c.startswith("nbhd_")]

    if not nbhd_cols:
        logger.info("No neighbourhood columns found — skipping ablation")
        return {}

    # With neighbourhood
    model_with = clone(model)
    scores_with = cross_val_score(
        model_with, X, y, cv=3, scoring="neg_mean_absolute_error", n_jobs=1
    )
    mae_with = float(-scores_with.mean())

    # Without neighbourhood
    X_without = X.drop(columns=nbhd_cols, errors="ignore")
    model_without = clone(model)
    scores_without = cross_val_score(
        model_without, X_without, y, cv=3, scoring="neg_mean_absolute_error", n_jobs=1
    )
    mae_without = float(-scores_without.mean())

    # Convert from log-scale MAE to approximate USD MAE
    result = {
        "mae_with_neighbourhood": round(mae_with, 4),
        "mae_without_neighbourhood": round(mae_without, 4),
        "absolute_degradation": round(mae_without - mae_with, 4),
        "relative_degradation_pct": round((mae_without - mae_with) / mae_with * 100, 2),
        "neighbourhood_columns_removed": nbhd_cols,
    }

    logger.info(
        "Feature ablation: MAE with nbhd=%.4f, without=%.4f (degradation=%.1f%%)",
        mae_with,
        mae_without,
        result["relative_degradation_pct"],
    )
    return result


# ===================================================================
# Fairness Summary
# ===================================================================


def generate_fairness_summary(
    group_biases: list[GroupBias],
    longo_results: list[LONGOResult],
    transfer_results: list[TransferResult],
) -> FairnessSummary:
    """Synthesise all bias findings into an executive summary.

    Assigns an overall risk level based on the number and severity
    of detected biases, and generates actionable recommendations.

    Args:
        group_biases: Results from compute_group_bias().
        longo_results: Results from cross_neighbourhood_cv().
        transfer_results: Results from cross_city_transfer().

    Returns:
        FairnessSummary with risk level and recommendations.
    """
    significant = [b for b in group_biases if b.is_significant]
    large_effect = [b for b in significant if b.cohens_d_magnitude in ("medium", "large")]

    total = len(group_biases)
    n_sig = len(significant)
    n_large = len(large_effect)

    # Risk assessment
    if n_large >= 3 or n_sig >= total * 0.4:
        risk = "HIGH"
    elif n_large >= 1 or n_sig >= total * 0.2:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # Recommendations
    recommendations: list[str] = []

    if n_sig > 0:
        recommendations.append(
            "Significant prediction biases detected. Consider stratified "
            "model training or sample reweighting for affected groups."
        )

    # Check geographic bias
    geo_biases = [b for b in significant if b.dimension in ("neighbourhood_group", "city")]
    if geo_biases:
        recommendations.append(
            "Geographic bias detected. The model systematically over/under-predicts "
            "for certain neighbourhoods. Consider regularising location features "
            "or using neighbourhood embeddings instead of one-hot encoding."
        )

    # Check price range bias
    price_biases = [b for b in significant if b.dimension == "price_quintile"]
    if price_biases:
        recommendations.append(
            "Price range bias detected. Consider training separate models for "
            "luxury vs budget segments, or using quantile regression for "
            "heteroscedastic prediction intervals."
        )

    # Check new listing cold-start
    cold_biases = [b for b in significant if b.dimension == "has_reviews"]
    if cold_biases:
        recommendations.append(
            "New listing cold-start bias detected. Listings without reviews "
            "have systematically different errors. Consider imputing review "
            "features with neighbourhood medians for new listings."
        )

    # Cross-city transfer
    cross_transfers = [t for t in transfer_results if t.train_city != t.eval_city]
    if cross_transfers:
        max_degrade = max(t.mae_degradation for t in cross_transfers)
        if max_degrade > 20:
            recommendations.append(
                f"Cross-city transfer shows up to ${max_degrade:.0f} MAE degradation. "
                "City-specific models are recommended over a single global model."
            )

    # LONGO gaps
    if longo_results:
        worst_gaps = sorted(longo_results, key=lambda r: r.gap_vs_cv, reverse=True)[:3]
        large_gap = [r for r in worst_gaps if r.gap_vs_cv > 10]
        if large_gap:
            areas = ", ".join(f"{r.city}/{r.neighbourhood_group}" for r in large_gap)
            recommendations.append(
                f"Neighbourhood generalisation gaps detected in: {areas}. "
                "The model may be memorising neighbourhood-specific patterns "
                "rather than learning generalisable pricing relationships."
            )

    if not recommendations:
        recommendations.append(
            "No critical biases detected. The model shows acceptable fairness "
            "across all tested dimensions."
        )

    top_biases = sorted(significant, key=lambda b: abs(b.cohens_d), reverse=True)[:5]

    return FairnessSummary(
        overall_risk=risk,
        total_biases_found=total,
        significant_biases=n_sig,
        top_biases=top_biases,
        recommendations=recommendations,
    )


# ===================================================================
# Report Saving
# ===================================================================


def save_bias_report(report: BiasAuditReport) -> Path:
    """Save the complete bias audit report as JSON and Markdown."""
    output_dir = OUTPUTS_DIR / report.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_data = {
        "experiment_id": report.experiment_id,
        "model_name": report.model_name,
        "fairness_summary": {
            "overall_risk": report.fairness_summary.overall_risk,
            "total_biases_tested": report.fairness_summary.total_biases_found,
            "significant_biases": report.fairness_summary.significant_biases,
            "recommendations": report.fairness_summary.recommendations,
        },
        "longo_results": [
            {
                "city": r.city,
                "neighbourhood_group": r.neighbourhood_group,
                "n_test": r.n_test,
                "mae": round(r.mae, 2),
                "mape": round(r.mape, 2) if not np.isnan(r.mape) else None,
                "mean_residual": round(r.mean_residual, 2),
                "gap_vs_cv": round(r.gap_vs_cv, 2),
            }
            for r in report.longo_results
        ],
        "transfer_matrix": [
            {
                "train_city": t.train_city,
                "eval_city": t.eval_city,
                "n_train": t.n_train,
                "n_eval": t.n_eval,
                "mae": round(t.mae, 2),
                "mape": round(t.mape, 2) if not np.isnan(t.mape) else None,
                "r2": round(t.r2, 4),
                "mae_degradation": round(t.mae_degradation, 2),
            }
            for t in report.transfer_matrix
        ],
        "group_biases": [
            {
                "dimension": b.dimension,
                "group": b.group,
                "n": b.n,
                "mean_residual": b.mean_residual,
                "ci": [b.ci_lower, b.ci_upper],
                "significant": b.is_significant,
                "cohens_d": b.cohens_d,
                "magnitude": b.cohens_d_magnitude,
                "direction": b.direction,
            }
            for b in report.group_biases
        ],
        "feature_ablation": report.feature_ablation,
    }

    with open(output_dir / "bias_audit_report.json", "w", encoding="utf-8") as fh:
        json.dump(json_data, fh, indent=2, default=str)

    # Markdown
    _generate_bias_markdown(report, output_dir)

    logger.info("Bias audit report saved: %s", output_dir)
    return output_dir


def _generate_bias_markdown(report: BiasAuditReport, output_dir: Path) -> None:
    """Generate a human-readable bias audit Markdown report."""
    fs = report.fairness_summary
    risk_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(fs.overall_risk, "⚪")

    lines = [
        f"# Bias Audit Report — `{report.experiment_id}`\n",
        "",
        f"**Model:** `{report.model_name}`  ",
        f"**Overall risk:** {risk_icon} **{fs.overall_risk}**  ",
        f"**Biases tested:** {fs.total_biases_found} | **Significant:** {fs.significant_biases}\n",
        "",
        "## Recommendations\n",
    ]
    for i, rec in enumerate(fs.recommendations, 1):
        lines.append(f"{i}. {rec}")

    # Transfer matrix
    if report.transfer_matrix:
        lines.extend(
            [
                "",
                "## Cross-City Transfer Matrix\n",
                "| Train → Eval | N Train | N Eval | MAE ($) | R² | Degradation |",
                "|:-------------|:--------|:-------|:--------|:---|:------------|",
            ]
        )
        for t in report.transfer_matrix:
            deg = f"+${t.mae_degradation:.0f}" if t.train_city != t.eval_city else "baseline"
            lines.append(
                f"| {t.train_city} → {t.eval_city} | {t.n_train:,} | {t.n_eval:,} "
                f"| ${t.mae:.0f} | {t.r2:.3f} | {deg} |"
            )

    # LONGO results
    if report.longo_results:
        lines.extend(
            [
                "",
                "## Neighbourhood Generalisation (LONGO)\n",
                "| City | Neighbourhood | N | MAE ($) | Bias ($) | Gap vs CV |",
                "|:-----|:-------------|:--|:--------|:---------|:----------|",
            ]
        )
        sorted_longo = sorted(report.longo_results, key=lambda r: r.gap_vs_cv, reverse=True)
        for r in sorted_longo[:20]:
            lines.append(
                f"| {r.city} | {r.neighbourhood_group} | {r.n_test} "
                f"| ${r.mae:.0f} | ${r.mean_residual:+.0f} | {r.gap_vs_cv:+.0f} |"
            )

    # Group biases
    sig_biases = [b for b in report.group_biases if b.is_significant]
    if sig_biases:
        lines.extend(
            [
                "",
                "## Significant Group Biases\n",
                "| Dimension | Group | N | Bias ($) | 95% CI | Cohen's d | Direction |",
                "|:----------|:------|:--|:---------|:-------|:----------|:----------|",
            ]
        )
        for b in sorted(sig_biases, key=lambda x: abs(x.cohens_d), reverse=True):
            lines.append(
                f"| {b.dimension} | {b.group} | {b.n:,} "
                f"| ${b.mean_residual:+.0f} | [${b.ci_lower:.0f}, ${b.ci_upper:.0f}] "
                f"| {b.cohens_d:.3f} ({b.cohens_d_magnitude}) | {b.direction} |"
            )

    # Feature ablation
    if report.feature_ablation:
        lines.extend(["", "## Neighbourhood Feature Ablation\n"])
        abl = report.feature_ablation
        lines.append(
            f"- MAE with neighbourhood features: {abl.get('mae_with_neighbourhood', 'N/A')}"
        )
        lines.append(f"- MAE without: {abl.get('mae_without_neighbourhood', 'N/A')}")
        lines.append(f"- Degradation: {abl.get('relative_degradation_pct', 'N/A')}%")
        cols = abl.get("neighbourhood_columns_removed", [])
        if cols:
            lines.append(f"- Columns removed: {', '.join(cols)}")

    with open(output_dir / "bias_audit_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ===================================================================
# Main Entry Point
# ===================================================================


def run_bias_audit(
    experiment_result: Any,
    feature_set: Any,
    split: Any,
    config: dict,
) -> BiasAuditReport:
    """Run the complete §6.4 bias and generalisation analysis.

    This is the main entry point. It:
      1. Runs LONGO cross-validation
      2. Runs cross-city transfer evaluation
      3. Computes group-level bias for all configured dimensions
      4. Runs neighbourhood feature ablation
      5. Generates a fairness summary with recommendations
      6. Saves all reports

    Args:
        experiment_result: ExperimentResult from train_experiment().
        feature_set: FeatureSet from build_feature_matrix().
        split: TrainTestSplit.
        config: Parsed ml_config.yaml dictionary.

    Returns:
        BiasAuditReport.
    """
    best_name = experiment_result.best_model_name
    best_trained = experiment_result.models[best_name]
    best_model = best_trained.model

    logger.info("Running bias audit for model: %s", best_name)

    # Baseline CV MAE for gap calculation
    baseline_mae = 0.0
    if best_trained.cv_result:
        baseline_mae = best_trained.cv_result.mean_metrics.get("mae", 0.0)

    # 1. Cross-neighbourhood (LONGO)
    longo = cross_neighbourhood_cv(
        best_model,
        feature_set.X,
        feature_set.y,
        feature_set.metadata_columns,
        config,
        baseline_mae,
    )

    # 2. Cross-city transfer
    transfer = cross_city_transfer(
        best_model,
        feature_set.X,
        feature_set.y,
        feature_set.metadata_columns,
        config,
    )

    # 3. Group-level bias on test set
    y_pred_test = best_trained.predict(split.X_test)
    biases = compute_group_bias(
        split.y_test.values,
        y_pred_test,
        split.meta_test,
        config,
    )

    # 4. Feature ablation
    ablation = neighbourhood_feature_ablation(
        best_model,
        feature_set.X,
        feature_set.y,
        config,
    )

    # 5. Fairness summary
    fairness = generate_fairness_summary(biases, longo, transfer)

    report = BiasAuditReport(
        experiment_id=experiment_result.experiment_id,
        model_name=best_name,
        longo_results=longo,
        transfer_matrix=transfer,
        group_biases=biases,
        fairness_summary=fairness,
        feature_ablation=ablation,
    )

    save_bias_report(report)
    return report
