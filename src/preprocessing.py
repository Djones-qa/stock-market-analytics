"""
preprocessing.py — Price data cleaning, adjustment, outlier handling.
"""

import pandas as pd
import numpy as np


def clean_price_data(df):
    """Validate and clean OHLCV price columns."""
    df = df.copy()
    price_cols = ["open", "high", "low", "close", "adj_close"]
    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            negatives = (df[col] < 0).sum()
            if negatives > 0:
                df.loc[df[col] < 0, col] = np.nan
                print(f"  {col}: removed {negatives} negative values")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
    return df


def validate_ohlc(df):
    """Ensure high >= low and high >= open/close."""
    df = df.copy()
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        invalid = (df["high"] < df["low"]) | (df["high"] < df["open"]) | (df["high"] < df["close"])
        count = invalid.sum()
        if count > 0:
            df.loc[invalid, ["open", "high", "low", "close"]] = np.nan
            print(f"  Flagged {count} rows with invalid OHLC relationships")
    return df


def handle_missing_prices(df, method="ffill"):
    """Fill missing prices using forward fill then backward fill."""
    df = df.copy()
    price_cols = [c for c in ["open", "high", "low", "close", "adj_close"] if c in df.columns]
    for col in price_cols:
        missing_before = df[col].isna().sum()
        if method == "ffill":
            df[col] = df[col].ffill().bfill()
        elif method == "interpolate":
            df[col] = df[col].interpolate(method="linear").bfill()
        filled = missing_before - df[col].isna().sum()
        if filled > 0:
            print(f"  {col}: filled {filled} missing values")
    return df


def remove_outliers_returns(df, price_col="close", threshold=0.5):
    """Remove days with extreme single-day returns (likely data errors)."""
    df = df.copy()
    if price_col in df.columns:
        returns = df[price_col].pct_change().abs()
        extreme = returns > threshold
        count = extreme.sum()
        if count > 0:
            df = df[~extreme]
            print(f"  Removed {count} extreme return days (>{threshold*100:.0f}%)")
    return df


def ensure_trading_dates(df, date_col="date"):
    """Sort by date and ensure proper datetime type."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df


def run_preprocessing_pipeline(df, date_col="date"):
    """Execute full price preprocessing pipeline."""
    print("Starting stock preprocessing pipeline...")
    print(f"  Input: {len(df):,} rows")
    df = ensure_trading_dates(df, date_col)
    df = clean_price_data(df)
    df = validate_ohlc(df)
    df = handle_missing_prices(df)
    df = remove_outliers_returns(df)
    print(f"  Output: {len(df):,} rows")
    print("Preprocessing complete.")
    return df
