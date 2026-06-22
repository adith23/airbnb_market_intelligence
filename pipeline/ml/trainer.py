"""Model training with cross-validation and experiment persistence.

Trains multiple model families (Ridge, Random Forest, XGBoost, LightGBM)
with hyperparameter tuning, records metrics, and saves model artifacts
to ``data/models/{experiment_id}/``.

Usage (from CLI via orchestrator):
    from pipeline.ml.trainer import train_experiment
    result = train_experiment(feature_set, split, config)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pipeline.utils import PROJECT_ROOT

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "data" / "models"


# ===================================================================
# Data Classes
# ===================================================================


@dataclass
class CVResult:
    """Cross-validation results for a single model."""

    fold_metrics: list[dict[str, float]]
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]


@dataclass
class TrainedModel:
    """A trained model with its metadata."""

    model: Any
    name: str
    best_params: dict[str, Any]
    cv_result: CVResult | None
    training_time_seconds: float
    feature_names: list[str]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


@dataclass
class ExperimentResult:
    """Result of a full training experiment."""

    experiment_id: str
    models: dict[str, TrainedModel]
    best_model_name: str
    best_metric_value: float
    primary_metric: str
    output_dir: Path
    config_snapshot: dict


# ===================================================================
# Experiment ID Generation
# ===================================================================


def _generate_experiment_id(config: dict) -> str:
    """Create a unique experiment ID from timestamp + config hash."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config_str = json.dumps(config, sort_keys=True, default=str)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return f"{timestamp}_{config_hash}"


# ===================================================================
# Model Instantiation
# ===================================================================


def _create_model(model_name: str, model_config: dict, params: dict | None = None) -> Any:
    """Instantiate a scikit-learn compatible model.

    Args:
        model_name: Key from config (ridge, random_forest, xgboost, lightgbm).
        model_config: Model section from ml_config.yaml.
        params: Optional hyperparameters to override defaults.

    Returns:
        Instantiated model object.
    """
    fixed = dict(model_config.get("fixed_params", {}))
    if params:
        fixed.update(params)

    # Replace YAML null with Python None for max_depth etc.
    fixed = {k: (None if v == "null" else v) for k, v in fixed.items()}

    if model_name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(**fixed)

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**fixed)

    if model_name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**fixed)

    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**fixed)

    raise ValueError(f"Unknown model family: {model_name}")


# ===================================================================
# Cross-Validation
# ===================================================================


def _compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, log_target: bool) -> dict[str, float]:
    """Compute regression metrics for a single fold."""
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )

    if log_target:
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred

    # Protect against negative predictions after inverse transform
    y_pred_orig = np.maximum(y_pred_orig, 0)

    # MAPE with protection against zero division
    nonzero = y_true_orig > 1.0
    if nonzero.sum() > 0:
        mape = float(np.mean(np.abs(y_true_orig[nonzero] - y_pred_orig[nonzero]) / y_true_orig[nonzero]) * 100)
    else:
        mape = np.nan

    return {
        "mae": float(mean_absolute_error(y_true_orig, y_pred_orig)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))),
        "mape": mape,
        "r2": float(r2_score(y_true_orig, y_pred_orig)),
        "median_ae": float(median_absolute_error(y_true_orig, y_pred_orig)),
    }


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
) -> CVResult:
    """Run stratified K-fold cross-validation.

    Args:
        model: Scikit-learn compatible estimator.
        X: Feature matrix.
        y: Target series (possibly log-transformed).
        config: Full ml_config dictionary.

    Returns:
        CVResult with per-fold and aggregate metrics.
    """
    from sklearn.model_selection import StratifiedKFold, KFold

    cv_cfg = config.get("cross_validation", {})
    n_folds = cv_cfg.get("n_folds", 5)
    random_state = cv_cfg.get("random_state", 42)
    log_target = config["target"].get("transform") == "log1p"

    # Try stratified; fall back to regular KFold if strat fails
    try:
        # Create quintile bins for stratification
        bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(cv.split(X, bins))
    except ValueError:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(cv.split(X))

    fold_metrics: list[dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_tr, y_tr)
        y_pred = fold_model.predict(X_val)

        metrics = _compute_fold_metrics(y_val.values, y_pred, log_target)
        fold_metrics.append(metrics)
        logger.debug("Fold %d: MAE=%.2f, R²=%.4f", fold_idx + 1, metrics["mae"], metrics["r2"])

    # Aggregate
    metric_keys = fold_metrics[0].keys()
    mean_m = {k: float(np.mean([f[k] for f in fold_metrics])) for k in metric_keys}
    std_m = {k: float(np.std([f[k] for f in fold_metrics])) for k in metric_keys}

    return CVResult(fold_metrics=fold_metrics, mean_metrics=mean_m, std_metrics=std_m)


# ===================================================================
# Hyperparameter Tuning
# ===================================================================


def _tune_hyperparameters(
    model_name: str,
    model_config: dict,
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
) -> tuple[Any, dict]:
    """Tune hyperparameters using grid or randomised search.

    Returns:
        (best_estimator, best_params) tuple.
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold

    cv_cfg = config.get("cross_validation", {})
    n_folds = cv_cfg.get("n_folds", 5)
    random_state = cv_cfg.get("random_state", 42)

    # Build param grid, replacing None values
    param_grid = {}
    for param, values in model_config.get("hyperparameter_grid", {}).items():
        param_grid[param] = [None if v is None else v for v in values]

    base_model = _create_model(model_name, model_config)
    tuning = model_config.get("tuning", "grid_search")

    try:
        bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    except ValueError:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        bins = None

    scoring = "neg_mean_absolute_error"

    if tuning == "grid_search":
        search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring=scoring,
            n_jobs=-1, refit=True, error_score="raise",
        )
    else:
        n_iter = model_config.get("tuning_iterations", 20)
        search = RandomizedSearchCV(
            base_model, param_grid, n_iter=n_iter, cv=cv,
            scoring=scoring, n_jobs=-1, random_state=random_state,
            refit=True, error_score="raise",
        )

    if bins is not None:
        search.fit(X, y, groups=None)
    else:
        search.fit(X, y)

    logger.info(
        "%s best params: %s (CV score: %.4f)",
        model_name, search.best_params_, search.best_score_,
    )
    return search.best_estimator_, search.best_params_


# ===================================================================
# Single Model Training
# ===================================================================


def train_single_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    feature_names: list[str],
) -> TrainedModel:
    """Train a single model with hyperparameter tuning + CV evaluation.

    Args:
        model_name: Key from config models section.
        X_train: Training features.
        y_train: Training target.
        config: Full ml_config dictionary.
        feature_names: List of feature column names.

    Returns:
        TrainedModel with fitted model and metrics.
    """
    model_config = config["models"][model_name]

    logger.info("Training %s...", model_name)
    start = time.time()

    # Tune hyperparameters
    best_model, best_params = _tune_hyperparameters(
        model_name, model_config, X_train, y_train, config
    )

    # Run CV with best model for detailed fold metrics
    cv_result = cross_validate_model(best_model, X_train, y_train, config)

    elapsed = time.time() - start
    logger.info(
        "%s trained in %.1fs — CV MAE: %.2f ± %.2f",
        model_name, elapsed,
        cv_result.mean_metrics["mae"], cv_result.std_metrics["mae"],
    )

    return TrainedModel(
        model=best_model,
        name=model_name,
        best_params=best_params,
        cv_result=cv_result,
        training_time_seconds=elapsed,
        feature_names=feature_names,
    )


# ===================================================================
# Quantile Regression
# ===================================================================


def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    feature_names: list[str],
) -> dict[str, TrainedModel]:
    """Train quantile regression models for prediction intervals.

    Uses LightGBM with ``objective='quantile'`` for the configured
    quantile levels (default: 10th and 90th percentile).

    Returns:
        Dict mapping ``q_{quantile}`` to TrainedModel.
    """
    qr_config = config.get("quantile_regression", {})
    if not qr_config.get("enabled", False):
        return {}

    quantiles = qr_config.get("quantiles", [0.10, 0.90])
    models = {}

    for q in quantiles:
        logger.info("Training quantile model (q=%.2f)...", q)
        start = time.time()

        try:
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            elapsed = time.time() - start

            models[f"q_{q:.2f}"] = TrainedModel(
                model=model,
                name=f"quantile_{q:.2f}",
                best_params={"alpha": q, "objective": "quantile"},
                cv_result=None,
                training_time_seconds=elapsed,
                feature_names=feature_names,
            )
            logger.info("Quantile model q=%.2f trained in %.1fs", q, elapsed)

        except ImportError:
            logger.warning("LightGBM not installed — skipping quantile model q=%.2f", q)

    return models


# ===================================================================
# Model Persistence
# ===================================================================


def save_experiment(result: ExperimentResult) -> Path:
    """Save all experiment artifacts to disk.

    Creates the directory ``data/models/{experiment_id}/`` and writes:
      - ``{model_name}.joblib`` for each trained model
      - ``feature_columns.json`` — ordered list of feature names
      - ``metrics.json`` — CV and summary metrics
      - ``config_snapshot.yaml`` — configuration used

    Args:
        result: ExperimentResult from train_experiment().

    Returns:
        Path to the experiment directory.
    """
    exp_dir = result.output_dir
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save each model
    for name, trained in result.models.items():
        model_path = exp_dir / f"{name}.joblib"
        joblib.dump(trained.model, model_path)
        logger.info("Saved model: %s", model_path)

    # Save feature columns
    feature_cols_path = exp_dir / "feature_columns.json"
    first_model = next(iter(result.models.values()))
    with open(feature_cols_path, "w", encoding="utf-8") as fh:
        json.dump(first_model.feature_names, fh, indent=2)

    # Save metrics
    metrics_data = {}
    for name, trained in result.models.items():
        entry: dict[str, Any] = {
            "best_params": {k: str(v) if v is not None else None for k, v in trained.best_params.items()},
            "training_time_seconds": round(trained.training_time_seconds, 2),
        }
        if trained.cv_result:
            entry["cv_mean"] = trained.cv_result.mean_metrics
            entry["cv_std"] = trained.cv_result.std_metrics
        metrics_data[name] = entry

    metrics_data["_best_model"] = result.best_model_name
    metrics_data["_primary_metric"] = result.primary_metric
    metrics_data["_best_metric_value"] = result.best_metric_value

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_data, fh, indent=2, default=str)

    # Save config snapshot
    import yaml
    with open(exp_dir / "config_snapshot.yaml", "w", encoding="utf-8") as fh:
        yaml.dump(result.config_snapshot, fh, default_flow_style=False)

    logger.info("Experiment saved: %s", exp_dir)
    return exp_dir


def load_experiment(experiment_id: str) -> tuple[dict[str, Any], list[str], Path]:
    """Load a saved experiment's models, feature columns, and metrics.

    Args:
        experiment_id: The experiment directory name.

    Returns:
        (models_dict, feature_columns, experiment_dir) tuple.
        models_dict maps model_name → loaded sklearn model.
    """
    exp_dir = MODELS_DIR / experiment_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment not found: {exp_dir}")

    # Load feature columns
    with open(exp_dir / "feature_columns.json", "r", encoding="utf-8") as fh:
        feature_columns = json.load(fh)

    # Load all model files
    models = {}
    for model_path in exp_dir.glob("*.joblib"):
        name = model_path.stem
        models[name] = joblib.load(model_path)
        logger.info("Loaded model: %s from %s", name, model_path)

    return models, feature_columns, exp_dir


# ===================================================================
# Main Entry Point
# ===================================================================


def train_experiment(
    feature_set: Any,
    split: Any,
    config: dict,
) -> ExperimentResult:
    """Train all enabled models and select the best one.

    This is the main entry point for training. It:
      1. Generates an experiment ID
      2. Trains each enabled model family
      3. Trains quantile models for prediction intervals
      4. Selects the best model by the primary metric
      5. Saves all artifacts to disk

    Args:
        feature_set: FeatureSet from build_feature_matrix().
        split: TrainTestSplit from prepare_train_test_split().
        config: Parsed ml_config.yaml dictionary.

    Returns:
        ExperimentResult with all trained models.
    """
    experiment_id = _generate_experiment_id(config)
    output_dir = MODELS_DIR / experiment_id
    logger.info("Starting experiment: %s", experiment_id)

    primary_metric = config["evaluation"].get("primary_metric", "mae")
    trained_models: dict[str, TrainedModel] = {}

    # Train each enabled model family
    for model_name, model_config in config["models"].items():
        if not model_config.get("enabled", True):
            logger.info("Skipping disabled model: %s", model_name)
            continue

        try:
            trained = train_single_model(
                model_name, split.X_train, split.y_train, config, feature_set.feature_names
            )
            trained_models[model_name] = trained
        except Exception as exc:
            logger.error("Failed to train %s: %s", model_name, exc)

    if not trained_models:
        raise RuntimeError("All model training attempts failed")

    # Train quantile models
    try:
        quantile_models = train_quantile_models(
            split.X_train, split.y_train, config, feature_set.feature_names
        )
        trained_models.update(quantile_models)
    except Exception as exc:
        logger.warning("Quantile model training failed: %s", exc)

    # Select best model (lower is better for MAE, RMSE, MAPE)
    lower_is_better = primary_metric in ("mae", "rmse", "mape", "median_ae")
    best_name = None
    best_value = float("inf") if lower_is_better else float("-inf")

    for name, trained in trained_models.items():
        if trained.cv_result is None:
            continue  # skip quantile models
        cv_val = trained.cv_result.mean_metrics.get(primary_metric)
        if cv_val is None:
            continue
        if lower_is_better and cv_val < best_value:
            best_value = cv_val
            best_name = name
        elif not lower_is_better and cv_val > best_value:
            best_value = cv_val
            best_name = name

    if best_name is None:
        best_name = next(iter(trained_models))
        best_value = 0.0

    logger.info(
        "Best model: %s (%s = %.4f)",
        best_name, primary_metric, best_value,
    )

    result = ExperimentResult(
        experiment_id=experiment_id,
        models=trained_models,
        best_model_name=best_name,
        best_metric_value=best_value,
        primary_metric=primary_metric,
        output_dir=output_dir,
        config_snapshot=config,
    )

    save_experiment(result)
    return result
