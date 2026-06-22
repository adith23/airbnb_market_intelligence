import pytest
import numpy as np
import pandas as pd

from pipeline.ml.bias_auditor import (
    _cohens_d,
    compute_group_bias,
    generate_fairness_summary
)

def test_cohens_d():
    group_resid = np.array([5, 6, 7, 8, 9])
    all_resid = np.array([1, 2, 3, 4, 5, 5, 6, 7, 8, 9])
    
    d, magnitude = _cohens_d(group_resid, all_resid)
    
    # group mean = 7, all mean = 5
    # The effect is large
    assert d > 0
    assert magnitude in ["medium", "large"]

def test_compute_group_bias():
    y_true_log = np.log1p(np.array([100]*15 + [200]*15))
    y_pred_log = np.log1p(np.array([110]*15 + [190]*15))
    
    metadata = pd.DataFrame({
        "city": ["Paris"]*15 + ["London"]*15
    })
    
    config = {
        "target": {"transform": "log1p"},
        "bias_audit": {
            "dimensions": [{"column": "city"}],
            "bootstrap_iterations": 100
        }
    }
    
    biases = compute_group_bias(y_true_log, y_pred_log, metadata, config)
    
    assert len(biases) == 2
    paris_bias = next(b for b in biases if b.group == "Paris")
    london_bias = next(b for b in biases if b.group == "London")
    
    # Paris true: 100, pred: 110, residual: -10 (over-predicted)
    assert paris_bias.direction == "over-predicted"
    assert paris_bias.mean_residual < 0
    
    # London true: 200, pred: 190, residual: +10 (under-predicted)
    assert london_bias.direction == "under-predicted"
    assert london_bias.mean_residual > 0
