import pytest
import numpy as np
import pandas as pd

from pipeline.ml.trainer import (
    _compute_fold_metrics,
    cross_validate_model,
    _generate_experiment_id
)

def test_generate_experiment_id():
    config1 = {"target": "price", "split": 0.2}
    config2 = {"target": "price", "split": 0.3}
    
    id1 = _generate_experiment_id(config1)
    id2 = _generate_experiment_id(config2)
    id1_again = _generate_experiment_id(config1)
    
    assert len(id1) > 8
    assert "_" in id1
    assert id1 != id2
    # The timestamp part might change, but the hash should be the same
    assert id1.split("_")[-1] == id1_again.split("_")[-1]

def test_compute_fold_metrics():
    y_true_log = np.log1p(np.array([100, 200, 300]))
    y_pred_log = np.log1p(np.array([100, 200, 300]))
    
    metrics = _compute_fold_metrics(y_true_log, y_pred_log, log_target=True)
    
    assert metrics["mae"] < 1e-5
    assert metrics["rmse"] < 1e-5
    assert metrics["mape"] < 1e-5
    assert metrics["r2"] == 1.0

def test_compute_fold_metrics_errors():
    y_true_log = np.log1p(np.array([100, 200]))
    y_pred_log = np.log1p(np.array([110, 180]))
    
    metrics = _compute_fold_metrics(y_true_log, y_pred_log, log_target=True)
    
    # errors: 10, 20. Mean absolute error = 15
    assert abs(metrics["mae"] - 15.0) < 1e-5
    
    # MAPE: 10/100 = 10%, 20/200 = 10%. Mean = 10%
    assert abs(metrics["mape"] - 10.0) < 1e-5

# A dummy model for CV testing
class DummyModel:
    def fit(self, X, y, groups=None):
        pass
        
    def predict(self, X):
        return np.log1p(np.ones(len(X)) * 150)

def test_cross_validate_model():
    X = pd.DataFrame({"feat": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    # y is all 150 except one to create some error
    y = pd.Series(np.log1p(np.array([150]*9 + [100])))
    
    config = {
        "cross_validation": {"n_folds": 2, "random_state": 42},
        "target": {"transform": "log1p"}
    }
    
    model = DummyModel()
    result = cross_validate_model(model, X, y, config)
    
    assert len(result.fold_metrics) == 2
    assert "mae" in result.mean_metrics
    assert "r2" in result.mean_metrics
    assert "mape" in result.mean_metrics
