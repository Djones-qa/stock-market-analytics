"""
portfolio_analysis.py — Portfolio performance, allocation, and risk metrics.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import sharpe_ratio, sortino_ratio, max_drawdown, value_at_risk  # noqa: E402


def portfolio_returns(df, weights=None, price_col="close"):
    """Calculate weighted portfolio returns."""
    if "ticker" not in df.columns:
        return pd.Series(dtype=float)
    tickers = df["ticker"].unique()
    if weights is None:
        weights = {t: 1.0 / len(tickers) for t in tickers}
    pivot = df.pivot_table(index="date", columns="ticker", values=price_col)
    returns = pivot.pct_change().dropna()
    weighted = sum(returns.get(t, 0) * w for t, w in weights.items() if t in returns.columns)
    return weighted


def portfolio_risk_report(returns):
    """Generate comprehensive portfolio risk metrics."""
    return {
        "total_return": round((1 + returns).prod() - 1, 4),
        "annualized_return": round(returns.mean() * 252, 4),
        "annualized_volatility": round(returns.std() * np.sqrt(252), 4),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": max_drawdown((1 + returns).cumprod()),
        "var_95": value_at_risk(returns, 0.95),
        "var_99": value_at_risk(returns, 0.99),
        "positive_days": round((returns > 0).mean(), 4),
        "best_day": round(returns.max(), 4),
        "worst_day": round(returns.min(), 4),
        "trading_days": len(returns),
    }


def correlation_between_stocks(df, price_col="close"):
    """Correlation matrix of returns between tickers."""
    if "ticker" not in df.columns:
        return pd.DataFrame()
    pivot = df.pivot_table(index="date", columns="ticker", values=price_col)
    return pivot.pct_change().corr().round(3)


def sector_allocation_analysis(df, weights=None):
    """Performance breakdown by sector."""
    if "sector" not in df.columns or "ticker" not in df.columns:
        return pd.DataFrame()
    ticker_sector = df[["ticker", "sector"]].drop_duplicates()
    tickers = df["ticker"].unique()
    if weights is None:
        weights = {t: 1.0 / len(tickers) for t in tickers}
    ticker_sector["weight"] = ticker_sector["ticker"].map(weights).fillna(0)
    return ticker_sector.groupby("sector")["weight"].sum().sort_values(ascending=False).round(4)


def rolling_sharpe(returns, window=60, risk_free_rate=0.045):
    """Calculate rolling Sharpe ratio."""
    excess = returns - risk_free_rate / 252
    rolling_mean = excess.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    return (rolling_mean / rolling_std.replace(0, np.nan) * np.sqrt(252)).round(4)
