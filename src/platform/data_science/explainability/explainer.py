"""SHAP-based model explainability for price prediction models.

Provides global feature importance, local (individual prediction)
explanations, and per-city importance breakdowns using SHAP values.

Supports TreeExplainer for tree-based models (XGBoost, LightGBM,
Random Forest) and KernelExplainer as a fallback for linear models.

Usage (from CLI via orchestrator):
    from src.platform.data_science.explainability.explainer import explain_model
    report = explain_model(experiment_result, split, config)
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


# ===================================================================
# Data Classes
# ===================================================================


@dataclass
class FeatureImportance:
    """Global feature importance from SHAP analysis."""

    feature: str
    mean_abs_shap: float
    rank: int


@dataclass
class LocalExplanation:
    """SHAP explanation for a single prediction."""

    listing_index: int
    true_value: float
    predicted_value: float
    top_positive: list[dict[str, float]]  # features pushing price UP
    top_negative: list[dict[str, float]]  # features pushing price DOWN
    base_value: float


@dataclass
class ExplainabilityReport:
    """Complete explainability report for a model."""

    experiment_id: str
    model_name: str
    global_importance: list[FeatureImportance]
    local_explanations: list[LocalExplanation]
    per_city_importance: dict[str, list[FeatureImportance]]
    shap_values_path: str | None
    n_samples_used: int


# ===================================================================
# SHAP Value Computation
# ===================================================================


def _detect_model_type(model: Any) -> str:
    """Detect whether a model is tree-based or linear.

    Returns:
        'tree' for tree-based models, 'linear' otherwise.
    """
    model_class = type(model).__name__.lower()
    tree_indicators = (
        "forest",
        "xgb",
        "lgbm",
        "lightgbm",
        "gradient",
        "tree",
        "extra",
    )
    return "tree" if any(t in model_class for t in tree_indicators) else "linear"


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    sample_size: int = 5000,
    random_state: int = 42,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Compute SHAP values for a model on a (sub)sample of data.

    For large datasets, subsampling is applied because SHAP computation
    is O(n × features × trees). TreeExplainer is used for tree models;
    a linear approximation is used for linear models.

    Args:
        model: Fitted scikit-learn compatible model.
        X: Feature matrix (full or subsample).
        sample_size: Max samples for SHAP computation.
        random_state: Seed for subsampling reproducibility.

    Returns:
        (shap_values, expected_value, X_sample) tuple.
        shap_values: (n_samples, n_features) array.
        expected_value: Scalar base value.
        X_sample: The (possibly subsampled) DataFrame used.
    """
    import shap

    # Subsample if needed
    if len(X) > sample_size:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X), size=sample_size, replace=False)
        X_sample = X.iloc[indices].copy()
    else:
        X_sample = X.copy()

    model_type = _detect_model_type(model)
    logger.info(
        "Computing SHAP values: model_type=%s, n_samples=%d, n_features=%d",
        model_type,
        len(X_sample),
        X_sample.shape[1],
    )

    if model_type == "tree":
        # Natively handles XGBoost without falling back to a callable masker
        explainer = shap.TreeExplainer(model)
        shap_values_obj = explainer(X_sample)
    else:
        # Use a small background dataset for models that require masking (e.g., linear)
        background = X_sample.sample(min(100, len(X_sample)), random_state=random_state)

        # Pass a callable to avoid "model is not callable" errors in default Explainer
        def predict_fn(x_array):
            df = pd.DataFrame(x_array, columns=X_sample.columns)
            for col in X_sample.columns:
                df[col] = df[col].astype(X_sample[col].dtype)
            preds = model.predict(df)
            return preds.flatten() if hasattr(preds, "flatten") else preds

        explainer = shap.Explainer(predict_fn, background)
        shap_values_obj = explainer(X_sample)

    # Safely extract numpy array (v2 Explanation object vs v1 numpy return)
    if hasattr(shap_values_obj, "values"):
        shap_array = shap_values_obj.values
    else:
        shap_array = np.asarray(shap_values_obj)

    # Safely extract expected value
    if (
        hasattr(shap_values_obj, "base_values")
        and shap_values_obj.base_values is not None
    ):
        expected_value = float(np.mean(shap_values_obj.base_values))
    elif hasattr(explainer, "expected_value") and explainer.expected_value is not None:
        expected_value = _safe_float(
            explainer.expected_value
            if np.isscalar(explainer.expected_value)
            else np.ravel(explainer.expected_value)[0]
        )
    else:
        expected_value = 0.0

    return shap_array, expected_value, X_sample


# ===================================================================
# Global Feature Importance
# ===================================================================


def global_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
) -> list[FeatureImportance]:
    """Rank features by mean absolute SHAP value.

    This provides a model-agnostic measure of feature importance
    that accounts for both direction and magnitude of each feature's
    contribution to predictions.

    Args:
        shap_values: (n_samples, n_features) SHAP array.
        feature_names: Ordered list of feature names.
        top_n: Number of top features to return.

    Returns:
        Sorted list of FeatureImportance, descending by importance.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    importances = []
    sorted_indices = np.argsort(mean_abs)[::-1]

    for rank, idx in enumerate(sorted_indices[:top_n], start=1):
        importances.append(
            FeatureImportance(
                feature=feature_names[idx],
                mean_abs_shap=round(float(mean_abs[idx]), 6),
                rank=rank,
            )
        )

    return importances


# ===================================================================
# Local Explanations
# ===================================================================


def _safe_float(val: Any) -> float:
    try:
        # Extract scalar from numpy arrays or pandas Series
        if hasattr(val, "item") and callable(val.item):
            try:
                val = val.item()
            except ValueError:
                pass  # Not a scalar array

        # Extract from list/tuple
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) > 0:
                val = val[0]
            else:
                return 0.0

        # Clean string representations
        if isinstance(val, str):
            val = val.strip("[]'\" ")

        return float(val)
    except Exception as e:
        return 0.0


def local_explanations(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    y_true: np.ndarray | None,
    y_pred: np.ndarray,
    base_value: float,
    n_explanations: int = 5,
    log_transformed: bool = True,
) -> list[LocalExplanation]:
    """Generate local explanations for individual predictions.

    Selects diverse predictions: best, worst, median, cheapest, most
    expensive.

    Args:
        shap_values: (n_samples, n_features) SHAP array.
        X_sample: Feature DataFrame (same rows as shap_values).
        y_true: True values (optional, for error-based selection).
        y_pred: Predicted values.
        base_value: SHAP expected value.
        n_explanations: Number of explanations to generate.
        log_transformed: Whether to inverse-transform for display.

    Returns:
        List of LocalExplanation dataclasses.
    """
    feature_names = list(X_sample.columns)
    n = min(n_explanations, len(X_sample))

    # Select diverse indices
    indices = set()
    indices.add(int(np.argmin(y_pred)))  # cheapest
    indices.add(int(np.argmax(y_pred)))  # most expensive
    if y_true is not None:
        errors = np.abs(y_true - y_pred)
        indices.add(int(np.argmax(errors)))  # worst prediction
        indices.add(int(np.argmin(errors)))  # best prediction
    indices.add(len(y_pred) // 2)  # median

    # Fill remaining with random
    rng = np.random.default_rng(42)
    while len(indices) < n:
        indices.add(int(rng.integers(0, len(y_pred))))

    explanations = []
    for idx in sorted(indices)[:n]:
        sv = shap_values[idx]
        sorted_feat_idx = np.argsort(sv)

        # Top 5 positive and negative
        top_pos = [
            {"feature": feature_names[fi], "shap_value": round(_safe_float(sv[fi]), 4)}
            for fi in sorted_feat_idx[::-1][:5]
            if sv[fi] > 0
        ]
        top_neg = [
            {"feature": feature_names[fi], "shap_value": round(_safe_float(sv[fi]), 4)}
            for fi in sorted_feat_idx[:5]
            if sv[fi] < 0
        ]

        true_val = _safe_float(y_true[idx]) if y_true is not None else np.nan
        pred_val = _safe_float(y_pred[idx])

        if log_transformed:
            true_val = float(np.expm1(true_val)) if not np.isnan(true_val) else np.nan
            pred_val = float(np.expm1(pred_val))

        explanations.append(
            LocalExplanation(
                listing_index=idx,
                true_value=round(true_val, 2),
                predicted_value=round(pred_val, 2),
                top_positive=top_pos,
                top_negative=top_neg,
                base_value=round(base_value, 4),
            )
        )

    return explanations


# ===================================================================
# Per-City Importance
# ===================================================================


def per_city_importance(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    meta_sample: pd.DataFrame,
    city_column: str = "city",
    top_n: int = 15,
) -> dict[str, list[FeatureImportance]]:
    """Compute separate feature importance rankings per city.

    Reveals whether the same features drive prices across cities
    or whether drivers differ structurally.

    Args:
        shap_values: (n_samples, n_features) SHAP array.
        X_sample: Feature DataFrame.
        meta_sample: Metadata DataFrame with city column.
        city_column: Column name for city grouping.
        top_n: Number of top features per city.

    Returns:
        Dict mapping city name to list of FeatureImportance.
    """
    feature_names = list(X_sample.columns)
    results = {}

    if city_column not in meta_sample.columns:
        logger.warning(
            "City column '%s' not in metadata — skipping per-city analysis", city_column
        )
        return results

    for city in meta_sample[city_column].unique():
        mask = meta_sample[city_column] == city
        city_shap = shap_values[mask.values]

        if len(city_shap) < 10:
            continue

        results[str(city)] = global_feature_importance(city_shap, feature_names, top_n)

    return results


# ===================================================================
# Report Saving
# ===================================================================


def save_explainability_report(report: ExplainabilityReport) -> Path:
    """Save explainability report as JSON and Markdown.

    Args:
        report: Complete ExplainabilityReport.

    Returns:
        Path to the output directory.
    """
    output_dir = OUTPUTS_DIR / report.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_data = {
        "experiment_id": report.experiment_id,
        "model_name": report.model_name,
        "n_samples_used": report.n_samples_used,
        "global_importance": [
            {"rank": fi.rank, "feature": fi.feature, "mean_abs_shap": fi.mean_abs_shap}
            for fi in report.global_importance
        ],
        "local_explanations": [
            {
                "listing_index": le.listing_index,
                "true_value": le.true_value,
                "predicted_value": le.predicted_value,
                "base_value": le.base_value,
                "top_positive_drivers": le.top_positive,
                "top_negative_drivers": le.top_negative,
            }
            for le in report.local_explanations
        ],
        "per_city_importance": {
            city: [
                {
                    "rank": fi.rank,
                    "feature": fi.feature,
                    "mean_abs_shap": fi.mean_abs_shap,
                }
                for fi in fis
            ]
            for city, fis in report.per_city_importance.items()
        },
    }

    with open(output_dir / "shap_report.json", "w", encoding="utf-8") as fh:
        json.dump(json_data, fh, indent=2, default=str)

    # Markdown
    _generate_shap_markdown(report, output_dir)

    logger.info("SHAP report saved: %s", output_dir)
    return output_dir


def _generate_shap_markdown(report: ExplainabilityReport, output_dir: Path) -> None:
    """Generate a human-readable SHAP report in Markdown."""
    lines = [
        f"# SHAP Explainability Report — `{report.experiment_id}`\n",
        "",
        f"**Model:** `{report.model_name}`  ",
        f"**Samples used:** {report.n_samples_used:,}\n",
        "",
        "## Global Feature Importance (Top 20)\n",
        "| Rank | Feature | Mean |SHAP| |",
        "|:-----|:--------|:------------|",
    ]

    for fi in report.global_importance[:20]:
        lines.append(f"| {fi.rank} | {fi.feature} | {fi.mean_abs_shap:.4f} |")

    # Per-city
    if report.per_city_importance:
        lines.extend(["", "## Per-City Feature Importance\n"])
        for city, fis in report.per_city_importance.items():
            lines.append(f"### {city}\n")
            lines.append("| Rank | Feature | Mean |SHAP| |")
            lines.append("|:-----|:--------|:------------|")
            for fi in fis[:10]:
                lines.append(f"| {fi.rank} | {fi.feature} | {fi.mean_abs_shap:.4f} |")
            lines.append("")

    # Local explanations
    if report.local_explanations:
        lines.extend(["", "## Local Explanations (Individual Predictions)\n"])
        for le in report.local_explanations:
            lines.append(
                f"### Listing #{le.listing_index} — True: ${le.true_value:,.0f}, "
                f"Predicted: ${le.predicted_value:,.0f}\n"
            )
            if le.top_positive:
                lines.append("**Price drivers (↑):**")
                for d in le.top_positive:
                    lines.append(f"- {d['feature']}: +{d['shap_value']:.4f}")
            if le.top_negative:
                lines.append("\n**Price reducers (↓):**")
                for d in le.top_negative:
                    lines.append(f"- {d['feature']}: {d['shap_value']:.4f}")
            lines.append("")

    with open(output_dir / "shap_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ===================================================================
# Main Entry Point
# ===================================================================


def explain_model(
    experiment_result: Any,
    split: Any,
    config: dict,
) -> ExplainabilityReport:
    """Generate a full SHAP explainability report for the best model.

    This is the main entry point for explainability. It:
      1. Selects the best model from the experiment
      2. Computes SHAP values on a subsample of test data
      3. Generates global, local, and per-city importance reports
      4. Saves all artifacts

    Args:
        experiment_result: ExperimentResult from train_experiment().
        split: TrainTestSplit.
        config: Parsed ml_config.yaml dictionary.

    Returns:
        ExplainabilityReport.
    """
    exp_cfg = config.get("explainability", {})
    sample_size = exp_cfg.get("shap_sample_size", 5000)
    top_n = exp_cfg.get("top_features", 20)
    n_local = exp_cfg.get("local_explanations", 5)
    random_state = exp_cfg.get("random_state", 42)
    log_transformed = config["target"].get("transform") == "log1p"

    best_name = experiment_result.best_model_name
    best_model = experiment_result.models[best_name]

    logger.info("Generating SHAP explanations for model: %s", best_name)

    # Compute SHAP values
    shap_vals, expected_val, X_sample = compute_shap_values(
        best_model.model, split.X_test, sample_size, random_state
    )

    # Predictions for the subsample
    y_pred_sample = np.asarray(best_model.predict(X_sample)).astype(float)

    # Match y_test to the subsample indices
    sample_indices = X_sample.index
    y_true_sample = (
        split.y_test.loc[sample_indices].values
        if hasattr(split.y_test, "loc")
        else None
    )

    # Match metadata to the subsample
    meta_sample = (
        split.meta_test.loc[sample_indices]
        if hasattr(split.meta_test, "loc")
        else split.meta_test
    )

    # Global importance
    global_imp = global_feature_importance(shap_vals, list(X_sample.columns), top_n)

    # Local explanations
    local_exp = local_explanations(
        shap_vals,
        X_sample,
        y_true_sample,
        y_pred_sample,
        expected_val,
        n_local,
        log_transformed,
    )

    # Per-city importance
    city_imp = per_city_importance(shap_vals, X_sample, meta_sample, "city", top_n)

    # Save SHAP values as Parquet
    shap_path = None
    try:
        output_dir = OUTPUTS_DIR / experiment_result.experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        shap_path_file = output_dir / "shap_values.parquet"
        import polars as pl

        shap_pl = pl.DataFrame(
            {col: shap_vals[:, i] for i, col in enumerate(X_sample.columns)}
        )
        shap_pl.write_parquet(shap_path_file)
        shap_path = str(shap_path_file)
    except Exception as exc:
        logger.warning("Could not save SHAP values to Parquet: %s", exc)

    report = ExplainabilityReport(
        experiment_id=experiment_result.experiment_id,
        model_name=best_name,
        global_importance=global_imp,
        local_explanations=local_exp,
        per_city_importance=city_imp,
        shap_values_path=shap_path,
        n_samples_used=len(X_sample),
    )

    save_explainability_report(report)
    return report
