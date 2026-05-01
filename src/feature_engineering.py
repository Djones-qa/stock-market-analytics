"""
feature_engineering.py — Technical indicators, returns, risk metrics, portfolio features.
"""

import pandas as pd
import numpy as np


def add_returns(df, price_col="close"):
    """Calculate daily, weekly, and monthly returns."""
    df = df.copy()
    df["daily_return"] = df[price_col].pct_change()
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1
    df["return_5d"] = df[price_col].pct_change(5)
    df["return_20d"] = df[price_col].pct_change(20)
    df["return_60d"] = df[price_col].pct_change(60)
    return df


def add_moving_averages(df, price_col="close", windows=None):
    """Add simple and exponential moving averages."""
    df = df.copy()
    if windows is None:
        windows = [20, 50, 100, 200]
    for w in windows:
        df[f"sma_{w}"] = df[price_col].rolling(w, min_periods=1).mean().round(2)
        df[f"ema_{w}"] = df[price_col].ewm(span=w, adjust=False).mean().round(2)
    if "sma_50" in df.columns and "sma_200" in df.columns:
        df["golden_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)
    return df


def add_rsi(df, price_col="close", period=14):
    """Calculate Relative Strength Index."""
    df = df.copy()
    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).round(2)
    df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
    df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
    return df


def add_macd(df, price_col="close", fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    df = df.copy()
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    df["macd_line"] = (ema_fast - ema_slow).round(4)
    df["macd_signal"] = df["macd_line"].ewm(span=signal, adjust=False).mean().round(4)
    df["macd_histogram"] = (df["macd_line"] - df["macd_signal"]).round(4)
    df["macd_bullish"] = ((df["macd_line"] > df["macd_signal"])
                          & (df["macd_line"].shift(1) <= df["macd_signal"].shift(1))).astype(int)
    return df


def add_bollinger_bands(df, price_col="close", period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    df = df.copy()
    sma = df[price_col].rolling(period).mean()
    std = df[price_col].rolling(period).std()
    df["bb_upper"] = (sma + std_dev * std).round(2)
    df["bb_middle"] = sma.round(2)
    df["bb_lower"] = (sma - std_dev * std).round(2)
    df["bb_width"] = ((df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]).round(4)
    df["bb_position"] = ((df[price_col] - df["bb_lower"])
                         / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)).round(4)
    return df


def add_volatility(df, price_col="close"):
    """Calculate rolling volatility metrics."""
    df = df.copy()
    if "daily_return" not in df.columns:
        df["daily_return"] = df[price_col].pct_change()
    df["volatility_20d"] = (df["daily_return"].rolling(20).std()
                            * np.sqrt(252)).round(4)
    df["volatility_60d"] = (df["daily_return"].rolling(60).std()
                            * np.sqrt(252)).round(4)
    df["atr"] = _average_true_range(df).round(2)
    return df


def _average_true_range(df, period=14):
    """Calculate Average True Range."""
    if not all(c in df.columns for c in ["high", "low", "close"]):
        return pd.Series(np.nan, index=df.index)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def add_volume_features(df):
    """Volume-based indicators."""
    df = df.copy()
    if "volume" not in df.columns:
        return df
    df["volume_sma_20"] = df["volume"].rolling(20, min_periods=1).mean().round(0)
    df["volume_ratio"] = (df["volume"] / df["volume_sma_20"].replace(0, np.nan)).round(2)
    df["is_high_volume"] = (df["volume_ratio"] > 1.5).astype(int)
    if "close" in df.columns:
        df["vwap_approx"] = ((df["close"] * df["volume"]).rolling(20).sum()
                             / df["volume"].rolling(20).sum()).round(2)
    return df


def add_temporal_features(df, date_col="date"):
    """Extract time components."""
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day_of_week"] = dt.dt.dayofweek
    df["quarter"] = dt.dt.quarter
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    df["is_quarter_end"] = dt.dt.is_quarter_end.astype(int)
    return df


def add_lag_features(df, price_col="close"):
    """Add price lag features for modeling."""
    df = df.copy()
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"price_lag_{lag}"] = df[price_col].shift(lag)
    for lag in [1, 5, 10]:
        df[f"return_lag_{lag}"] = df["daily_return"].shift(lag) if "daily_return" in df.columns else np.nan
    return df


def run_feature_pipeline(df):
    """Execute full feature engineering pipeline."""
    print("Starting feature engineering...")
    initial = len(df.columns)
    df = add_returns(df)
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volatility(df)
    df = add_volume_features(df)
    df = add_temporal_features(df)
    df = add_lag_features(df)
    new_cols = len(df.columns) - initial
    print(f"  Added {new_cols} features ({initial} -> {len(df.columns)} total)")
    print("Feature engineering complete.")
    return df
