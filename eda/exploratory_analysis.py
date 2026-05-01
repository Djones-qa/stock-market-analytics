"""
exploratory_analysis.py — Automated EDA for stock market data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import format_currency, format_pct, sharpe_ratio, max_drawdown


def price_summary(df, price_col="close"):
    stats = df[price_col].describe().round(2)
    stats["skew"] = round(df[price_col].skew(), 3)
    stats["kurtosis"] = round(df[price_col].kurtosis(), 3)
    return stats


def return_statistics(df):
    if "daily_return" not in df.columns:
        return {}
    r = df["daily_return"].dropna()
    return {
        "mean_daily": round(r.mean(), 6),
        "std_daily": round(r.std(), 6),
        "annualized_return": round(r.mean() * 252, 4),
        "annualized_volatility": round(r.std() * np.sqrt(252), 4),
        "sharpe": sharpe_ratio(r),
        "max_drawdown": max_drawdown(df["close"]) if "close" in df.columns else None,
        "best_day": round(r.max(), 4),
        "worst_day": round(r.min(), 4),
        "positive_days_pct": round((r > 0).mean(), 4),
    }


def ticker_comparison(df, price_col="close"):
    if "ticker" not in df.columns:
        return pd.DataFrame()
    return df.groupby("ticker").agg(
        first_price=(price_col, "first"),
        last_price=(price_col, "last"),
        avg_volume=("volume", "mean") if "volume" in df.columns else (price_col, "count"),
        observations=(price_col, "count"),
    ).round(2)


def sector_breakdown(df):
    if "sector" not in df.columns:
        return pd.DataFrame()
    return df.groupby("sector").agg(
        tickers=("ticker", "nunique") if "ticker" in df.columns else ("close", "count"),
        avg_return=("daily_return", "mean") if "daily_return" in df.columns else ("close", "count"),
        avg_volume=("volume", "mean") if "volume" in df.columns else ("close", "count"),
    ).round(4)


def correlation_matrix(df):
    numeric = df.select_dtypes(include=[np.number])
    key_cols = [c for c in ["close","volume","daily_return","rsi","macd_line",
                            "volatility_20d","bb_position"] if c in numeric.columns]
    if key_cols:
        return numeric[key_cols].corr().round(3)
    return numeric.corr().round(3)


def monthly_performance(df, price_col="close"):
    if "year" not in df.columns or "month" not in df.columns:
        return pd.DataFrame()
    monthly = df.groupby(["year","month"])[price_col].agg(["first","last"])
    monthly["return"] = ((monthly["last"] - monthly["first"]) / monthly["first"]).round(4)
    return monthly


def run_full_eda(df):
    print("\n=== Price Summary ===")
    print(price_summary(df).to_string())
    print("\n=== Return Statistics ===")
    rs = return_statistics(df)
    for k, v in rs.items():
        print(f"  {k}: {v}")
    print("\n=== Sector Breakdown ===")
    sb = sector_breakdown(df)
    if not sb.empty:
        print(sb.to_string())
    print("\nEDA complete.")
