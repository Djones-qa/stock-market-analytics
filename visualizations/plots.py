"""
plots.py — 8 professional stock market visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = "visualizations/output"


def _save(fig, name, output_dir=None):
    out = output_dir or OUTPUT_DIR
    Path(out).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}/{name}", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}/{name}")
    plt.close(fig)


def plot_price_history(df, price_col="close", output_dir=None):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(pd.to_datetime(df["date"]), df[price_col], linewidth=1.5,
            color="#1976D2", label="Close", alpha=0.9)
    for sma, color in [("sma_50","#FF9800"),("sma_200","#F44336")]:
        if sma in df.columns:
            ax.plot(pd.to_datetime(df["date"]), df[sma], linewidth=1.2,
                    color=color, label=sma.upper(), alpha=0.7)
    ax.set_title("Price History with Moving Averages", fontweight="bold", fontsize=14)
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "01_price_history.png", output_dir)


def plot_return_distribution(df, output_dir=None):
    if "daily_return" not in df.columns:
        return
    returns = df["daily_return"].dropna()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(returns, bins=100, density=True, color="#4CAF50",
            alpha=0.7, edgecolor="white", label="Actual")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(returns.mean(), color="red", linestyle="--", label=f"Mean: {returns.mean():.4f}")
    ax.set_title("Daily Return Distribution", fontweight="bold", fontsize=14)
    ax.set_xlabel("Daily Return")
    ax.legend()
    _save(fig, "02_return_distribution.png", output_dir)


def plot_volatility_trend(df, output_dir=None):
    if "volatility_20d" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(16, 6))
    dates = pd.to_datetime(df["date"])
    if "volatility_60d" in df.columns:
        ax.fill_between(dates, df["volatility_60d"], alpha=0.2, color="#F44336")
    ax.plot(dates, df["volatility_20d"], color="#F44336", linewidth=1.5, label="20-day Vol")
    ax.set_title("Rolling Annualized Volatility", fontweight="bold", fontsize=14)
    ax.set_ylabel("Volatility")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "03_volatility_trend.png", output_dir)


def plot_rsi_chart(df, output_dir=None):
    if "rsi" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(16, 5))
    dates = pd.to_datetime(df["date"])
    ax.plot(dates, df["rsi"], color="#9C27B0", linewidth=1.2)
    ax.axhline(70, color="red", linestyle="--", alpha=0.7, label="Overbought (70)")
    ax.axhline(30, color="green", linestyle="--", alpha=0.7, label="Oversold (30)")
    ax.fill_between(dates, 30, 70, alpha=0.05, color="gray")
    ax.set_title("RSI (14-Period)", fontweight="bold", fontsize=14)
    ax.set_ylabel("RSI")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "04_rsi_chart.png", output_dir)


def plot_bollinger_bands(df, price_col="close", output_dir=None):
    if "bb_upper" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(16, 7))
    dates = pd.to_datetime(df["date"])
    ax.plot(dates, df[price_col], color="#1976D2", linewidth=1.5, label="Close")
    ax.plot(dates, df["bb_upper"], color="#F44336", linewidth=1, alpha=0.7, label="Upper BB")
    ax.plot(dates, df["bb_lower"], color="#4CAF50", linewidth=1, alpha=0.7, label="Lower BB")
    ax.fill_between(dates, df["bb_lower"], df["bb_upper"], alpha=0.08, color="#9C27B0")
    ax.set_title("Bollinger Bands", fontweight="bold", fontsize=14)
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "05_bollinger_bands.png", output_dir)


def plot_volume_analysis(df, output_dir=None):
    if "volume" not in df.columns:
        return
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [2, 1]})
    dates = pd.to_datetime(df["date"])
    axes[0].plot(dates, df["close"], color="#1976D2", linewidth=1.5)
    axes[0].set_title("Price and Volume", fontweight="bold", fontsize=14)
    axes[0].set_ylabel("Price ($)")
    axes[0].grid(alpha=0.3)
    colors = ["#4CAF50" if r >= 0 else "#F44336"
              for r in df["daily_return"].fillna(0)] if "daily_return" in df.columns else ["#2196F3"] * len(df)
    axes[1].bar(dates, df["volume"], color=colors, alpha=0.7, width=1)
    if "volume_sma_20" in df.columns:
        axes[1].plot(dates, df["volume_sma_20"], color="#FF9800", linewidth=1.5, label="20d Avg")
        axes[1].legend()
    axes[1].set_ylabel("Volume")
    plt.tight_layout()
    _save(fig, "06_volume_analysis.png", output_dir)


def plot_monthly_returns_heatmap(df, output_dir=None):
    if "year" not in df.columns or "month" not in df.columns:
        return
    monthly = df.groupby(["year","month"])["daily_return"].sum() if "daily_return" in df.columns else None
    if monthly is None:
        return
    pivot = (monthly * 100).unstack()
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, cmap="RdYlGn", center=0, annot=True, fmt=".1f",
                ax=ax, linewidths=0.5, cbar_kws={"label": "Return (%)"})
    ax.set_title("Monthly Returns Heatmap (%)", fontweight="bold", fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    _save(fig, "07_monthly_returns_heatmap.png", output_dir)


def plot_drawdown(df, price_col="close", output_dir=None):
    fig, ax = plt.subplots(figsize=(16, 6))
    dates = pd.to_datetime(df["date"])
    prices = df[price_col]
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak * 100
    ax.fill_between(dates, drawdown, 0, color="#F44336", alpha=0.4)
    ax.plot(dates, drawdown, color="#F44336", linewidth=1)
    ax.set_title("Drawdown from Peak", fontweight="bold", fontsize=14)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.3)
    _save(fig, "08_drawdown.png", output_dir)


def generate_all_plots(df, output_dir=None):
    out = output_dir or OUTPUT_DIR
    print(f"Generating all plots to {out}/...")
    plot_price_history(df, output_dir=out)
    plot_return_distribution(df, output_dir=out)
    plot_volatility_trend(df, output_dir=out)
    plot_rsi_chart(df, output_dir=out)
    plot_bollinger_bands(df, output_dir=out)
    plot_volume_analysis(df, output_dir=out)
    plot_monthly_returns_heatmap(df, output_dir=out)
    plot_drawdown(df, output_dir=out)
    print("All 8 plots generated.")
