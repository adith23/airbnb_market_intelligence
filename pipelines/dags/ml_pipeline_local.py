"""Orchestrator for the ML pipeline (§6.1 and §6.4).

Coordinates feature engineering, training, evaluation, explainability,
and bias auditing. Integrates with the existing pipeline automation
framework to record metadata in DuckDB.

Usage (from CLI):
    from pipelines.dags.ml_pipeline_local import run_ml_pipeline
    result = run_ml_pipeline(config_path, force=False)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.platform.common.metadata import complete_run, fail_run, start_run
from src.platform.common.utils import get_db_path
from src.platform.data_science.validation.bias_auditor import run_bias_audit
from src.platform.data_science.validation.evaluator import evaluate_experiment
from src.platform.data_science.explainability.explainer import explain_model
from src.platform.data_science.training.trainer import train_experiment
from src.platform.feature_engineering.feature_store import (
    build_feature_matrix,
    load_ml_config,
    prepare_train_test_split,
)

logger = logging.getLogger(__name__)


@dataclass
class MLPipelineResult:
    """Summary of a full ML pipeline execution."""

    experiment_id: str
    best_model: str
    mae: float
    r2: float
    bias_risk: str
    success: bool
    error: str | None = None


def _tracked_ml_stage(
    stage_name: str,
    work_fn: callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run an ML stage with DuckDB metadata tracking."""
    run_id = start_run(
        city="multi-city",
        stage=f"ml_{stage_name}",
        source_file="airbnb.duckdb",
    )

    try:
        result = work_fn(*args, **kwargs)
        complete_run(run_id)
        return result
    except Exception as exc:
        fail_run(run_id, str(exc))
        raise RuntimeError(f"ML stage '{stage_name}' failed: {exc}") from exc


def run_ml_pipeline(
    config_path: str | Path | None = None,
    force: bool = False,
) -> MLPipelineResult:
    """Run the complete end-to-end ML pipeline.

    Stages:
      1. Feature Engineering
      2. Model Training & CV
      3. Evaluation (Test Set)
      4. Explainability (SHAP)
      5. Bias Audit (§6.4)

    Args:
        config_path: Path to ml_config.yaml.
        force: Retrain even if unchanged (passed to stages if needed).

    Returns:
        MLPipelineResult summary.
    """
    logger.info("Starting ML Pipeline Execution")

    try:
        # Load config
        config = load_ml_config(config_path)

        # 1. Feature Engineering
        logger.info("\n" + "=" * 40 + "\n[1/5] FEATURE ENGINEERING\n" + "=" * 40)
        feature_set = _tracked_ml_stage(
            "feature_engineering",
            build_feature_matrix,
            config,
            get_db_path(),
        )
        split = prepare_train_test_split(feature_set, config)

        # 2. Training
        logger.info("\n" + "=" * 40 + "\n[2/5] MODEL TRAINING\n" + "=" * 40)
        experiment = _tracked_ml_stage(
            "train",
            train_experiment,
            feature_set,
            split,
            config,
        )

        # 3. Evaluation
        logger.info("\n" + "=" * 40 + "\n[3/5] EVALUATION\n" + "=" * 40)
        eval_report = _tracked_ml_stage(
            "evaluate",
            evaluate_experiment,
            experiment,
            split,
            config,
        )

        # 4. Explainability
        logger.info("\n" + "=" * 40 + "\n[4/5] EXPLAINABILITY (SHAP)\n" + "=" * 40)
        # Using a lambda to avoid passing kwargs directly to explainer
        # or we just pass the args correctly
        explain_report = _tracked_ml_stage(
            "explain", lambda: explain_model(experiment, split, config)
        )

        # 5. Bias Audit
        logger.info("\n" + "=" * 40 + "\n[5/5] BIAS AUDIT\n" + "=" * 40)
        bias_report = _tracked_ml_stage(
            "bias_audit",
            run_bias_audit,
            experiment,
            feature_set,
            split,
            config,
        )

        # Final Summary
        best_name = experiment.best_model_name
        metrics = eval_report.test_metrics[best_name]

        return MLPipelineResult(
            experiment_id=experiment.experiment_id,
            best_model=best_name,
            mae=metrics.mae,
            r2=metrics.r2,
            bias_risk=bias_report.fairness_summary.overall_risk,
            success=True,
        )

    except Exception as exc:
        logger.error("ML Pipeline failed: %s", exc)
        return MLPipelineResult(
            experiment_id="FAILED",
            best_model="NONE",
            mae=0.0,
            r2=0.0,
            bias_risk="UNKNOWN",
            success=False,
            error=str(exc),
        )
