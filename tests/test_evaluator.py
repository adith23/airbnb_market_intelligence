import numpy as np
import pandas as pd

from src.platform.data_science.evaluation.evaluator import (
    compute_metrics,
    stratified_error_analysis,
)


def test_compute_metrics():
    # log scale true/pred
    y_true_log = np.log1p(np.array([100, 200, 300]))
    y_pred_log = np.log1p(np.array([110, 190, 300]))

    metrics = compute_metrics(y_true_log, y_pred_log, model_name="test", log_transformed=True)

    # Original values: 100, 200, 300
    # Pred values: 110, 190, 300
    # Absolute errors: 10, 10, 0
    # MAE = 20 / 3 = 6.666...
    assert metrics.model_name == "test"
    assert abs(metrics.mae - 6.6666) < 1e-3
    assert metrics.n_samples == 3


def test_stratified_error_analysis():
    y_true_log = np.log1p(np.array([100, 100, 200, 200]))
    y_pred_log = np.log1p(np.array([110, 90, 220, 180]))

    # Groups: A, A, B, B
    groups = pd.DataFrame({"city": ["Paris", "Paris", "London", "London"]})

    # min_group_size=1 so we can test with small data
    results = stratified_error_analysis(
        y_true_log,
        y_pred_log,
        groups,
        dimensions=["city"],
        log_transformed=True,
        min_group_size=1,
    )

    assert len(results) == 2

    paris_res = next(r for r in results if r.group == "Paris")
    # Paris true: 100, 100; pred: 110, 90; residuals (true - pred): -10, +10
    assert abs(paris_res.mae - 10.0) < 1e-4
    assert abs(paris_res.mean_residual - 0.0) < 1e-4
