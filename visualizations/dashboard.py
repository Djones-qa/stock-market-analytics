"""
dashboard.py — Interactive Plotly stock market dashboards exported as HTML.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def create_price_dashboard(df):
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=("Price History", "Daily Returns",
                        "Volume Trend", "RSI"),
        vertical_spacing=0.12)
    dates = pd.to_datetime(df["date"])
    fig.add_trace(go.Scatter(x=dates, y=df["close"],
        mode="lines", line=dict(color="#1976D2", width=1.5),
        name="Close"), row=1, col=1)
    if "sma_50" in df.columns:
        fig.add_trace(go.Scatter(x=dates, y=df["sma_50"],
            mode="lines", line=dict(color="#FF9800", width=1),
            name="SMA 50"), row=1, col=1)
    if "daily_return" in df.columns:
        fig.add_trace(go.Histogram(x=df["daily_return"].dropna(),
            nbinsx=80, marker_color="#4CAF50",
            name="Returns"), row=1, col=2)
    if "volume" in df.columns:
        fig.add_trace(go.Bar(x=dates, y=df["volume"],
            marker_color="#2196F3", opacity=0.6,
            name="Volume"), row=2, col=1)
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=dates, y=df["rsi"],
            mode="lines", line=dict(color="#9C27B0", width=1.2),
            name="RSI"), row=2, col=2)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=2)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=2)
    fig.update_layout(title="Stock Price Dashboard", height=700,
                      showlegend=False, template="plotly_white")
    return fig


def create_performance_dashboard(df):
    if "cumulative_return" not in df.columns:
        return None
    fig = make_subplots(rows=2, cols=1,
        subplot_titles=("Cumulative Return", "Rolling Volatility"))
    dates = pd.to_datetime(df["date"])
    fig.add_trace(go.Scatter(x=dates, y=df["cumulative_return"] * 100,
        mode="lines", fill="tozeroy",
        line=dict(color="#4CAF50", width=2),
        name="Cumulative Return %"), row=1, col=1)
    if "volatility_20d" in df.columns:
        fig.add_trace(go.Scatter(x=dates, y=df["volatility_20d"],
            mode="lines", line=dict(color="#F44336", width=1.5),
            name="20d Vol"), row=2, col=1)
    fig.update_layout(title="Performance Dashboard", height=600,
                      showlegend=False, template="plotly_white")
    return fig


def export_dashboards(df, output_dir="visualizations/output"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("Generating interactive dashboards...")
    fig1 = create_price_dashboard(df)
    fig1.write_html(str(out / "dashboard_price.html"))
    print(f"  Saved: {out}/dashboard_price.html")
    fig2 = create_performance_dashboard(df)
    if fig2:
        fig2.write_html(str(out / "dashboard_performance.html"))
        print(f"  Saved: {out}/dashboard_performance.html")
    print("Dashboards complete.")
