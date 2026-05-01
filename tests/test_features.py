"""Unit tests for stock feature engineering."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.feature_engineering import (
    add_returns, add_moving_averages, add_rsi,
    add_macd, add_bollinger_bands, add_volatility,
    add_volume_features, add_temporal_features
)


@pytest.fixture
def sample_stock_df():
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range("2023-01-01", periods=n)
    price = 100 + np.cumsum(np.random.normal(0.05, 1.5, n))
    return pd.DataFrame({
        "date": dates,
        "open": price - np.random.uniform(0, 1, n),
        "high": price + np.random.uniform(0, 2, n),
        "low": price - np.random.uniform(0, 2, n),
        "close": price.round(2),
        "volume": np.random.randint(500000, 5000000, n),
    })


class TestReturns:
    def test_adds_daily_return(self, sample_stock_df):
        result = add_returns(sample_stock_df)
        assert "daily_return" in result.columns

    def test_adds_cumulative_return(self, sample_stock_df):
        result = add_returns(sample_stock_df)
        assert "cumulative_return" in result.columns

    def test_adds_log_return(self, sample_stock_df):
        result = add_returns(sample_stock_df)
        assert "log_return" in result.columns


class TestMovingAverages:
    def test_adds_sma_columns(self, sample_stock_df):
        result = add_moving_averages(sample_stock_df)
        assert "sma_20" in result.columns
        assert "sma_50" in result.columns
        assert "sma_200" in result.columns

    def test_adds_ema_columns(self, sample_stock_df):
        result = add_moving_averages(sample_stock_df)
        assert "ema_20" in result.columns

    def test_adds_golden_cross(self, sample_stock_df):
        result = add_moving_averages(sample_stock_df)
        assert "golden_cross" in result.columns


class TestRSI:
    def test_adds_rsi(self, sample_stock_df):
        result = add_rsi(sample_stock_df)
        assert "rsi" in result.columns

    def test_rsi_range(self, sample_stock_df):
        result = add_rsi(sample_stock_df)
        valid = result["rsi"].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100


class TestMACD:
    def test_adds_macd(self, sample_stock_df):
        result = add_macd(sample_stock_df)
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns


class TestBollingerBands:
    def test_adds_bands(self, sample_stock_df):
        result = add_bollinger_bands(sample_stock_df)
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_middle" in result.columns


class TestVolatility:
    def test_adds_volatility(self, sample_stock_df):
        df = add_returns(sample_stock_df)
        result = add_volatility(df)
        assert "volatility_20d" in result.columns


class TestVolume:
    def test_adds_volume_ratio(self, sample_stock_df):
        result = add_volume_features(sample_stock_df)
        assert "volume_ratio" in result.columns
        assert "is_high_volume" in result.columns


class TestTemporal:
    def test_adds_year_month(self, sample_stock_df):
        result = add_temporal_features(sample_stock_df)
        assert "year" in result.columns
        assert "month" in result.columns
        assert "day_of_week" in result.columns
