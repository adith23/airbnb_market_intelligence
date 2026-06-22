"""ML pipeline subpackage for §6.1 Price Prediction + §6.4 Bias Audit.

Modules:
    feature_store   — Build reproducible feature matrices from DuckDB
    trainer         — Train and cross-validate multiple model families
    evaluator       — Compute metrics, residual analysis, model comparison
    explainer       — SHAP-based model explainability
    bias_auditor    — Cross-neighbourhood, cross-city, fairness analysis
    orchestrator    — Coordinate all ML stages with metadata tracking
"""
