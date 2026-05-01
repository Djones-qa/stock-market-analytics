"""
utils.py — Shared utilities: config, formatting, risk calculations.
"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent


def load_config(config_path=None):
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_currency(value):
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def format_pct(value, decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_large_number(value):
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1e12:
        return f"${value/1e12:.1f}T"
    if abs(value) >= 1e9:
        return f"${value/1e9:.1f}B"
    if abs(value) >= 1e6:
        return f"${value/1e6:.1f}M"
    return f"${value:,.0f}"


def sharpe_ratio(returns, risk_free_rate=0.045, periods=252):
    excess = returns - risk_free_rate / periods
    if excess.std() == 0:
        return 0
    return round((excess.mean() / excess.std()) * np.sqrt(periods), 4)


def sortino_ratio(returns, risk_free_rate=0.045, periods=252):
    excess = returns - risk_free_rate / periods
    downside = returns[returns < 0].std()
    if downside == 0:
        return 0
    return round((excess.mean() / downside) * np.sqrt(periods), 4)


def max_drawdown(prices):
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return round(drawdown.min(), 4)


def value_at_risk(returns, confidence=0.95):
    return round(returns.quantile(1 - confidence), 4)


def dataset_summary(df):
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_pct": round(missing_cells / total_cells * 100, 2) if total_cells else 0,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "duplicates": df.duplicated().sum(),
    }


def print_summary(df, label="Dataset"):
    info = dataset_summary(df)
    print(f"\n{'='*50}")
    print(f"  {label} Summary")
    print(f"{'='*50}")
    print(f"  Rows:        {info['rows']:,}")
    print(f"  Columns:     {info['columns']}")
    print(f"  Missing:     {info['missing_pct']}%")
    print(f"  Duplicates:  {info['duplicates']:,}")
    print(f"  Memory:      {info['memory_mb']} MB")
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
        print(f"  Date Range:  {dates.min().date()} to {dates.max().date()}")
    if "ticker" in df.columns:
        print(f"  Tickers:     {df['ticker'].nunique()}")
    print(f"{'='*50}\n")


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)
