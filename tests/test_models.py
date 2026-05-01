"""Unit tests for model evaluation utilities."""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.evaluate import compute_metrics
from src.utils import sharpe_ratio, sortino_ratio, max_drawdown, value_at_risk


@pytest.fixture
def sample_predictions():
    np.random.seed(42)
    y_true = np.random.uniform(100, 200, 100)
    y_pred = y_true + np.random.normal(0, 3, 100)
    return y_true, y_pred


class TestComputeMetrics:
    def test_returns_all_keys(self, sample_predictions):
        y_true, y_pred = sample_predictions
        metrics = compute_metrics(y_true, y_pred)
        expected = {"MAE","RMSE","MAPE","R2","Median_AE","n_samples"}
        assert set(metrics.keys()) == expected

    def test_mae_positive(self, sample_predictions):
        y_true, y_pred = sample_predictions
        assert compute_metrics(y_true, y_pred)["MAE"] > 0

    def test_perfect_prediction(self):
        y = np.array([100, 150, 200])
        metrics = compute_metrics(y, y)
        assert metrics["MAE"] == 0
        assert metrics["R2"] == 1.0

    def test_sample_count(self, sample_predictions):
        y_true, y_pred = sample_predictions
        assert compute_metrics(y_true, y_pred)["n_samples"] == 100


class TestRiskMetrics:
    def test_sharpe_ratio(self):
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        sr = sharpe_ratio(returns)
        assert isinstance(sr, float)

    def test_max_drawdown_negative(self):
        prices = pd.Series([100, 110, 90, 95, 80, 100])
        mdd = max_drawdown(prices)
        assert mdd < 0

    def test_max_drawdown_no_loss(self):
        prices = pd.Series([100, 110, 120, 130])
        mdd = max_drawdown(prices)
        assert mdd == 0

    def test_value_at_risk(self):
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        var = value_at_risk(returns, 0.95)
        assert var < 0

    def test_sortino_ratio(self):
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        sr = sortino_ratio(returns)
        assert isinstance(sr, float)


import pandas as pd
