"""Model evaluation, residual analysis, and comparison reporting.

Computes test-set metrics, stratified error analysis across multiple
grouping dimensions, residual diagnostics, and generates both JSON
and Markdown reports.

Usage (from CLI via orchestrator):
    from src.platform.data_science.validation.evaluator import evaluate_experiment
    report = evaluate_experiment(experiment, split, config)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.platform.common.utils import OUTPUT_DIR

logger = logging.getLogger(__name__)

OUTPUTS_DIR = OUTPUT_DIR / "ml"


# Data Classes
@dataclass
class ModelMetrics:
    """Evaluation metrics for a single model on a dataset split."""

    model_name: str
    mae: float
    rmse: float
    mape: float
    r2: float
    median_ae: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "mape": round(self.mape, 4),
            "r2": round(self.r2, 4),
            "median_ae": round(self.median_ae, 4),
            "n_samples": self.n_samples,
        }


@dataclass
class ResidualDiagnostics:
    """Residual distribution diagnostics for a model."""

    mean_residual: float
    std_residual: float
    skewness: float
    kurtosis: float
    normality_p: float  # Jarque-Bera p-value
    heteroscedasticity_p: float  # Breusch-Pagan proxy


@dataclass
class StratifiedError:
    """Error metrics for a single group in a stratification dimension."""

    dimension: str
    group: str
    n: int
    mae: float
    mape: float
    mean_residual: float  # Positive = under-predicted, Negative = over-predicted
    median_residual: float


@dataclass
class EvaluationReport:
    """Complete evaluation report for an experiment."""

    experiment_id: str
    test_metrics: dict[str, ModelMetrics]
    best_model_name: str
    model_comparison: pd.DataFrame
    stratified_errors: list[StratifiedError]
    residual_diagnostics: dict[str, ResidualDiagnostics]
    prediction_intervals: dict[str, Any] | None
    success_check: dict[str, bool]


# Core Metrics
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    log_transformed: bool = True,
) -> ModelMetrics:
    """Compute regression metrics on the original price scale.

    When the target was log-transformed during training, predictions
    are inverse-transformed via ``expm1()`` before metric computation.

    Args:
        y_true: True target values (possibly log-scale).
        y_pred: Predicted values (same scale as y_true).
        model_name: Name for labelling.
        log_transformed: Whether to apply expm1() inverse transform.

    Returns:
        ModelMetrics with MAE, RMSE, MAPE, R², MedAE.
    """
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )

    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if log_transformed:
        y_t = np.expm1(y_t)
        y_p = np.expm1(y_p)

    y_p = np.maximum(y_p, 0)

    # MAPE with zero-protection
    nonzero = y_t > 1.0
    if nonzero.sum() > 0:
        mape = float(np.mean(np.abs(y_t[nonzero] - y_p[nonzero]) / y_t[nonzero]) * 100)
    else:
        mape = np.nan

    return ModelMetrics(
        model_name=model_name,
        mae=float(mean_absolute_error(y_t, y_p)),
        rmse=float(np.sqrt(mean_squared_error(y_t, y_p))),
        mape=mape,
        r2=float(r2_score(y_t, y_p)),
        median_ae=float(median_absolute_error(y_t, y_p)),
        n_samples=len(y_t),
    )


# Residual Diagnostics
def compute_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    log_transformed: bool = True,
) -> ResidualDiagnostics:
    """Analyse the distribution of prediction residuals.

    Computes normality (Jarque-Bera) and a proxy for heteroscedasticity
    (correlation between |residual| and predicted value).

    Args:
        y_true: True values.
        y_pred: Predicted values.
        log_transformed: Whether to inverse-transform first.

    Returns:
        ResidualDiagnostics with distribution summary.
    """
    from scipy import stats as sp_stats

    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if log_transformed:
        y_t = np.expm1(y_t)
        y_p = np.expm1(y_p)

    y_p = np.maximum(y_p, 0)
    residuals = y_t - y_p

    # Normality test
    try:
        _, normality_p = sp_stats.jarque_bera(residuals)
    except Exception:
        normality_p = np.nan

    # Heteroscedasticity proxy: correlation between |residual| and predicted
    try:
        corr, hetero_p = sp_stats.spearmanr(np.abs(residuals), y_p)
    except Exception:
        hetero_p = np.nan

    return ResidualDiagnostics(
        mean_residual=float(np.mean(residuals)),
        std_residual=float(np.std(residuals)),
        skewness=float(sp_stats.skew(residuals)),
        kurtosis=float(sp_stats.kurtosis(residuals)),
        normality_p=float(normality_p),
        heteroscedasticity_p=float(hetero_p),
    )


# Stratified Error Analysis
def stratified_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups_df: pd.DataFrame,
    dimensions: list[str],
    log_transformed: bool = True,
    min_group_size: int = 30,
) -> list[StratifiedError]:
    """Compute MAE/MAPE per group across multiple grouping dimensions.

    This is the core function for bias detection — it reveals whether
    errors are systematic by location, property type, or price range.

    Args:
        y_true: True target values.
        y_pred: Predicted values.
        groups_df: DataFrame with grouping columns (city, room_type, etc.).
        dimensions: Column names to stratify by.
        log_transformed: Whether to inverse-transform first.
        min_group_size: Skip groups smaller than this.

    Returns:
        List of StratifiedError, one per (dimension, group) pair.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if log_transformed:
        y_t = np.expm1(y_t)
        y_p = np.expm1(y_p)

    y_p = np.maximum(y_p, 0)
    residuals = y_t - y_p

    results: list[StratifiedError] = []

    for dim in dimensions:
        if dim not in groups_df.columns:
            logger.warning("Stratification dimension '%s' not found in metadata", dim)
            continue

        for group_val, idxs in groups_df.groupby(dim).groups.items():
            n = len(idxs)
            if n < min_group_size:
                continue

            mask = groups_df.index.isin(idxs)
            g_true = y_t[mask]
            g_pred = y_p[mask]
            g_resid = residuals[mask]

            # MAPE
            nonzero = g_true > 1.0
            if nonzero.sum() > 0:
                mape = float(
                    np.mean(np.abs(g_true[nonzero] - g_pred[nonzero]) / g_true[nonzero])
                    * 100
                )
            else:
                mape = np.nan

            results.append(
                StratifiedError(
                    dimension=dim,
                    group=str(group_val),
                    n=n,
                    mae=float(np.mean(np.abs(g_true - g_pred))),
                    mape=mape,
                    mean_residual=float(np.mean(g_resid)),
                    median_residual=float(np.median(g_resid)),
                )
            )

    return results


# Prediction Intervals
def evaluate_prediction_intervals(
    quantile_models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    log_transformed: bool = True,
) -> dict[str, Any]:
    """Evaluate prediction interval coverage and width.

    Args:
        quantile_models: Dict of {name: model} for quantile regressors.
        X_test: Test features.
        y_test: Test target.
        log_transformed: Whether to inverse-transform.

    Returns:
        Dict with coverage, mean width, and calibration metrics.
    """
    if not quantile_models:
        return None

    y_t = y_test.values.copy()
    if log_transformed:
        y_t = np.expm1(y_t)

    predictions = {}
    for name, model in quantile_models.items():
        pred = model.predict(X_test)
        if log_transformed:
            pred = np.expm1(pred)
        predictions[name] = np.maximum(pred, 0)

    # Find lower and upper quantile predictions
    names = sorted(predictions.keys())
    if len(names) < 2:
        return None

    lower_pred = predictions[names[0]]
    upper_pred = predictions[names[-1]]

    # Coverage: fraction of true values within [lower, upper]
    coverage = float(np.mean((y_t >= lower_pred) & (y_t <= upper_pred)))
    mean_width = float(np.mean(upper_pred - lower_pred))
    median_width = float(np.median(upper_pred - lower_pred))

    return {
        "lower_quantile": names[0],
        "upper_quantile": names[-1],
        "coverage": round(coverage, 4),
        "mean_interval_width": round(mean_width, 2),
        "median_interval_width": round(median_width, 2),
        "n_samples": len(y_t),
    }


# Model Comparison
def compare_models(test_metrics: dict[str, ModelMetrics]) -> pd.DataFrame:
    """Create a side-by-side model comparison table.

    Args:
        test_metrics: {model_name: ModelMetrics} from test-set evaluation.

    Returns:
        DataFrame with models as rows and metrics as columns, sorted
        by MAE ascending.
    """
    records = [m.to_dict() for m in test_metrics.values()]
    df = pd.DataFrame(records).set_index("model_name")
    return df.sort_values("mae")


# Report Generation
def _check_success_thresholds(
    metrics: ModelMetrics,
    thresholds: dict[str, float],
) -> dict[str, bool]:
    """Check if the best model meets success criteria."""
    checks = {}
    for metric_name, threshold in thresholds.items():
        actual = getattr(metrics, metric_name, None)
        if actual is None:
            checks[metric_name] = False
            continue

        if metric_name in ("mae", "rmse", "mape", "median_ae"):
            checks[metric_name] = actual <= threshold
        else:  # r2
            checks[metric_name] = actual >= threshold

    return checks


def generate_evaluation_report(
    experiment_id: str,
    test_metrics: dict[str, ModelMetrics],
    best_model_name: str,
    stratified_errors: list[StratifiedError],
    residual_diag: dict[str, ResidualDiagnostics],
    config: dict,
    prediction_intervals: dict[str, Any] | None = None,
) -> EvaluationReport:
    """Assemble the complete evaluation report.

    Args:
        experiment_id: Experiment identifier.
        test_metrics: Test-set metrics per model.
        best_model_name: Name of the selected best model.
        stratified_errors: Error analysis by group.
        residual_diag: Residual diagnostics per model.
        config: ML config for success thresholds.
        prediction_intervals: Optional PI evaluation results.

    Returns:
        EvaluationReport dataclass.
    """
    comparison = compare_models(test_metrics)
    thresholds = config.get("evaluation", {}).get("success_thresholds", {})
    success = _check_success_thresholds(test_metrics[best_model_name], thresholds)

    return EvaluationReport(
        experiment_id=experiment_id,
        test_metrics=test_metrics,
        best_model_name=best_model_name,
        model_comparison=comparison,
        stratified_errors=stratified_errors,
        residual_diagnostics=residual_diag,
        prediction_intervals=prediction_intervals,
        success_check=success,
    )


# Saving Reports
def save_evaluation_report(report: EvaluationReport) -> Path:
    """Save evaluation report as JSON and Markdown.

    Args:
        report: Complete EvaluationReport.

    Returns:
        Path to the output directory.
    """
    output_dir = OUTPUTS_DIR / report.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    json_data = {
        "experiment_id": report.experiment_id,
        "best_model": report.best_model_name,
        "success_thresholds_met": report.success_check,
        "test_metrics": {name: m.to_dict() for name, m in report.test_metrics.items()},
        "residual_diagnostics": {
            name: {
                "mean_residual": round(d.mean_residual, 4),
                "std_residual": round(d.std_residual, 4),
                "skewness": round(d.skewness, 4),
                "kurtosis": round(d.kurtosis, 4),
                "normality_p": round(d.normality_p, 6),
                "heteroscedasticity_p": round(d.heteroscedasticity_p, 6),
            }
            for name, d in report.residual_diagnostics.items()
        },
        "stratified_errors": [
            {
                "dimension": e.dimension,
                "group": e.group,
                "n": e.n,
                "mae": round(e.mae, 2),
                "mape": round(e.mape, 2) if not np.isnan(e.mape) else None,
                "mean_residual": round(e.mean_residual, 2),
                "median_residual": round(e.median_residual, 2),
            }
            for e in report.stratified_errors
        ],
        "prediction_intervals": report.prediction_intervals,
    }

    with open(output_dir / "evaluation_report.json", "w", encoding="utf-8") as fh:
        json.dump(json_data, fh, indent=2, default=str)

    # Stratified errors as Parquet for downstream analysis
    if report.stratified_errors:
        import polars as pl

        strat_pl = pl.DataFrame(
            [
                {
                    "dimension": e.dimension,
                    "group": e.group,
                    "n": e.n,
                    "mae": e.mae,
                    "mape": e.mape,
                    "mean_residual": e.mean_residual,
                    "median_residual": e.median_residual,
                }
                for e in report.stratified_errors
            ]
        )
        strat_pl.write_parquet(output_dir / "stratified_errors.parquet")

    # Markdown report
    _generate_markdown_report(report, output_dir)

    logger.info("Evaluation report saved: %s", output_dir)
    return output_dir


def _generate_markdown_report(report: EvaluationReport, output_dir: Path) -> None:
    """Generate a human-readable Markdown evaluation report."""
    lines = [
        f"# Evaluation Report — Experiment `{report.experiment_id}`\n",
        "",
        f"**Best model:** `{report.best_model_name}`\n",
        "",
        "## Success Criteria\n",
        "| Metric | Threshold | Met? |",
        "|:-------|:----------|:-----|",
    ]

    for metric, met in report.success_check.items():
        icon = "✅" if met else "❌"
        best_m = report.test_metrics.get(report.best_model_name)
        actual = getattr(best_m, metric, "N/A") if best_m else "N/A"
        lines.append(f"| {metric} | — | {icon} ({actual:.4f}) |")

    lines.extend(["", "## Model Comparison (Test Set)\n", ""])
    lines.append(report.model_comparison.to_markdown())

    # Residual diagnostics
    lines.extend(["", "## Residual Diagnostics\n", ""])
    for name, diag in report.residual_diagnostics.items():
        lines.append(f"### {name}\n")
        lines.append(f"- Mean residual: ${diag.mean_residual:.2f}")
        lines.append(f"- Std residual: ${diag.std_residual:.2f}")
        lines.append(f"- Skewness: {diag.skewness:.4f}")
        lines.append(f"- Normality p-value: {diag.normality_p:.4e}")
        lines.append(f"- Heteroscedasticity p-value: {diag.heteroscedasticity_p:.4e}")
        lines.append("")

    # Stratified errors summary
    if report.stratified_errors:
        lines.extend(["## Stratified Error Analysis\n", ""])
        strat_df = pd.DataFrame(
            [
                {
                    "Dim": e.dimension,
                    "Group": e.group,
                    "N": e.n,
                    "MAE": f"${e.mae:.2f}",
                    "MAPE": f"{e.mape:.1f}%",
                    "Bias": f"${e.mean_residual:.2f}",
                }
                for e in report.stratified_errors
            ]
        )
        lines.append(strat_df.to_markdown(index=False))

    # Prediction intervals
    if report.prediction_intervals:
        lines.extend(["", "## Prediction Intervals\n", ""])
        pi = report.prediction_intervals
        lines.append(f"- Coverage: {pi['coverage']:.1%}")
        lines.append(f"- Mean interval width: ${pi['mean_interval_width']:.2f}")
        lines.append(f"- Median interval width: ${pi['median_interval_width']:.2f}")

    with open(output_dir / "evaluation_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# Main Entry Point
def evaluate_experiment(
    experiment_result: Any,
    split: Any,
    config: dict,
) -> EvaluationReport:
    """Evaluate all models on the held-out test set.

    This is the main entry point for evaluation. It:
      1. Predicts on the test set with each model
      2. Computes metrics on the original price scale
      3. Runs stratified error analysis
      4. Computes residual diagnostics
      5. Evaluates prediction intervals (if quantile models exist)
      6. Saves reports

    Args:
        experiment_result: ExperimentResult from train_experiment().
        split: TrainTestSplit.
        config: Parsed ml_config.yaml dictionary.

    Returns:
        EvaluationReport.
    """
    log_transformed = config["target"].get("transform") == "log1p"
    test_metrics: dict[str, ModelMetrics] = {}
    residual_diag: dict[str, ResidualDiagnostics] = {}
    quantile_models: dict[str, Any] = {}

    for name, trained in experiment_result.models.items():
        if trained.cv_result is None:
            # Quantile model — handle separately
            quantile_models[name] = trained.model
            continue

        y_pred = trained.predict(split.X_test)
        metrics = compute_metrics(split.y_test.values, y_pred, name, log_transformed)
        test_metrics[name] = metrics

        diag = compute_residual_diagnostics(
            split.y_test.values, y_pred, log_transformed
        )
        residual_diag[name] = diag

        logger.info(
            "%s test: MAE=$%.2f, MAPE=%.1f%%, R²=%.4f",
            name,
            metrics.mae,
            metrics.mape,
            metrics.r2,
        )

    # Stratified error analysis for the best model
    best_name = experiment_result.best_model_name
    best_model = experiment_result.models[best_name]
    y_pred_best = best_model.predict(split.X_test)

    strat_dims = [
        d["column"] for d in config.get("bias_audit", {}).get("dimensions", [])
    ]
    if not strat_dims:
        strat_dims = ["city", "room_type", "neighbourhood_group", "price_quintile"]

    stratified = stratified_error_analysis(
        split.y_test.values,
        y_pred_best,
        split.meta_test,
        strat_dims,
        log_transformed,
    )

    # Prediction intervals
    pi_result = evaluate_prediction_intervals(
        quantile_models, split.X_test, split.y_test, log_transformed
    )

    report = generate_evaluation_report(
        experiment_result.experiment_id,
        test_metrics,
        best_name,
        stratified,
        residual_diag,
        config,
        pi_result,
    )

    save_evaluation_report(report)
    return report
