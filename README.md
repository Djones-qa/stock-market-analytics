# Stock Market Analytics

**Price trend analysis, portfolio optimization, and stock forecasting with Python and ML.**

[![CI](https://github.com/Djones-qa/stock-market-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/Djones-qa/stock-market-analytics/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Project Overview

End-to-end stock market analytics pipeline that processes historical price data, engineers technical indicators, analyzes portfolio risk, and forecasts prices using multiple ML models. Demonstrates time-series analysis, quantitative finance, and data engineering skills.

### Key Objectives

- **Price Trend Analysis** - Moving averages, momentum, and trend detection
- **Technical Indicators** - RSI, MACD, Bollinger Bands, ATR, volume analysis
- **Portfolio Analytics** - Sharpe ratio, Sortino ratio, max drawdown, VaR
- **ML Forecasting** - Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost
- **Risk Assessment** - Volatility modeling, drawdown analysis, sector diversification
- **Trading Signals** - MA crossover, RSI, MACD, Bollinger Band composite signals
- **Sector Analysis** - Performance comparison across GICS sectors
- **8 Professional Visualizations** - Price charts, heatmaps, and interactive Plotly dashboards

---

## Repository Structure

    stock-market-analytics/
    .github/workflows/ci.yml
    config/config.yaml
    data/
        raw/                        Original price CSVs
        processed/                  Cleaned data with indicators
        external/                   Benchmarks, economic data
        README.md
    eda/
        exploratory_analysis.py     Price stats, return analysis, correlations
        portfolio_analysis.py       Portfolio risk, allocation, rolling Sharpe
        technical_signals.py        MA crossover, RSI, MACD, BB signals
    models/
        train.py                    5-model comparison with time-series CV
        predict.py                  Single and batch forecasting
        evaluate.py                 MAE, RMSE, MAPE, residual analysis
        saved_models/
    notebooks/
        01_data_exploration.ipynb
        02_preprocessing_features.ipynb
        03_model_training.ipynb
        04_visualization.ipynb
    sql/
        create_tables.sql           prices, holdings, indicators, forecasts
        price_analysis.sql          Monthly/yearly performance, 52-week range
        portfolio_queries.sql       Holdings value, sector allocation, P&L
        technical_queries.sql       RSI extremes, golden crosses, volatility
    src/
        __init__.py
        data_loader.py
        preprocessing.py            OHLCV cleaning, outlier removal
        feature_engineering.py      Technical indicators, returns, lag features
        utils.py                    Risk metrics, formatting utilities
    tests/
        test_preprocessing.py       10 data cleaning tests
        test_features.py            16 technical indicator tests
        test_models.py              9 evaluation and risk metric tests
    visualizations/
        plots.py                    8 matplotlib/seaborn charts
        dashboard.py                Interactive Plotly dashboards
        output/
    .gitignore
    requirements.txt
    README.md

---

## Analytics Pipeline

    Raw Prices -> Validate -> Clean -> Technical Indicators -> Analyze -> Forecast -> Visualize

### Preprocessing (src/preprocessing.py)
- OHLCV validation (negative prices, high < low checks)
- Forward-fill missing prices with configurable methods
- Extreme return outlier removal (>50% single-day moves)
- Date sorting and trading day alignment

### Feature Engineering (src/feature_engineering.py)
- Returns: daily, log, cumulative, 5/20/60-day
- Moving Averages: SMA and EMA (20, 50, 100, 200), golden cross detection
- RSI: 14-period with overbought/oversold flags
- MACD: 12/26/9 with histogram and bullish crossover signals
- Bollinger Bands: 20-period, 2 std dev, width and position metrics
- Volatility: 20/60-day annualized, ATR
- Volume: 20-day SMA, volume ratio, high volume flags, VWAP approximation
- Lag features: 1-20 day price and return lookbacks

### Forecasting Models (models/train.py)

| Model | Approach |
|---|---|
| Linear Regression | Baseline linear model |
| Ridge | L2 regularized regression |
| Random Forest | 200-tree ensemble |
| Gradient Boosting | 200 sequential estimators |
| XGBoost | Gradient boosting with regularization |

All models use TimeSeriesSplit (5-fold) to prevent data leakage.

### Risk Metrics (src/utils.py)
- **Sharpe Ratio** - Risk-adjusted return vs risk-free rate
- **Sortino Ratio** - Downside risk-adjusted return
- **Max Drawdown** - Largest peak-to-trough decline
- **Value at Risk (VaR)** - 95th and 99th percentile loss estimates

---

## Visualizations

### Static Plots (8 charts)
1. Price history with SMA 50/200 overlay
2. Daily return distribution with mean marker
3. Rolling annualized volatility
4. RSI with overbought/oversold zones
5. Bollinger Bands with price channel
6. Price and volume analysis (dual panel)
7. Monthly returns heatmap (year x month)
8. Drawdown from peak

### Interactive Dashboards (Plotly HTML)
- Price dashboard (4-panel: price, returns, volume, RSI)
- Performance dashboard (cumulative return, rolling volatility)

---

## Quick Start

    git clone https://github.com/Djones-qa/stock-market-analytics.git
    cd stock-market-analytics
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pytest tests/ -v --cov=src

Place your CSV in data/raw/ then run notebooks 01 through 04.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Analysis | pandas, NumPy, SciPy, statsmodels |
| Technical | ta (Technical Analysis Library) |
| ML | scikit-learn, XGBoost, LightGBM |
| Database | SQLite, SQLAlchemy |
| Visualization | matplotlib, seaborn, Plotly |
| Testing | pytest, pytest-cov |
| CI/CD | GitHub Actions |

---

## Author

**Darrius Jones**
QA Automation Specialist | Backend Engineering | Financial Data Analytics
