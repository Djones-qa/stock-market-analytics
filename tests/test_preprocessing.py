"""Unit tests for stock data preprocessing."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.preprocessing import (
    clean_price_data, validate_ohlc, handle_missing_prices,
    remove_outliers_returns, ensure_trading_dates
)


class TestCleanPriceData:
    def test_removes_negative_prices(self):
        df = pd.DataFrame({"close": [100, -5, 110], "open": [99, 98, 109]})
        result = clean_price_data(df)
        assert result["close"].isna().sum() == 1

    def test_keeps_valid_prices(self):
        df = pd.DataFrame({"close": [100, 105, 110]})
        result = clean_price_data(df)
        assert result["close"].isna().sum() == 0

    def test_cleans_volume(self):
        df = pd.DataFrame({"volume": ["1000", None, "5000"]})
        result = clean_price_data(df)
        assert result["volume"].dtype == int


class TestValidateOHLC:
    def test_flags_high_below_low(self):
        df = pd.DataFrame({
            "open": [100], "high": [90], "low": [95], "close": [92]
        })
        result = validate_ohlc(df)
        assert result["high"].isna().sum() == 1

    def test_valid_ohlc_passes(self):
        df = pd.DataFrame({
            "open": [100], "high": [110], "low": [95], "close": [105]
        })
        result = validate_ohlc(df)
        assert result["high"].isna().sum() == 0


class TestHandleMissing:
    def test_fills_gaps(self):
        df = pd.DataFrame({"close": [100, np.nan, np.nan, 106]})
        result = handle_missing_prices(df)
        assert result["close"].isna().sum() == 0

    def test_forward_fills(self):
        df = pd.DataFrame({"close": [100, np.nan, 105]})
        result = handle_missing_prices(df, method="ffill")
        assert result["close"].iloc[1] == 100


class TestOutlierRemoval:
    def test_removes_extreme_returns(self):
        prices = [100] + [101] * 98 + [300]
        df = pd.DataFrame({"close": prices})
        result = remove_outliers_returns(df, threshold=0.5)
        assert len(result) < len(df)


class TestDateSorting:
    def test_sorts_by_date(self):
        df = pd.DataFrame({
            "date": ["2024-03-01", "2024-01-01", "2024-02-01"],
            "close": [150, 140, 145]
        })
        result = ensure_trading_dates(df)
        assert result["close"].iloc[0] == 140
