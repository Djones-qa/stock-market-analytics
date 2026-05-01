# Data Directory

## Structure

| Folder | Purpose |
|---|---|
| `raw/` | Original stock price CSVs, API downloads |
| `processed/` | Cleaned data with technical indicators and features |
| `external/` | Market indices, sector ETFs, economic indicators |

## Expected Schema

| Column | Type | Description |
|---|---|---|
| `date` | date | Trading date |
| `ticker` | string | Stock ticker symbol |
| `open` | float | Opening price |
| `high` | float | Intraday high |
| `low` | float | Intraday low |
| `close` | float | Closing price |
| `adj_close` | float | Adjusted closing price |
| `volume` | int | Shares traded |
| `sector` | string | GICS sector classification |
| `market_cap` | float | Market capitalization |
| `pe_ratio` | float | Price-to-earnings ratio |
| `dividend_yield` | float | Annual dividend yield |
| `beta` | float | Beta vs benchmark |

## Data Sources

- **Primary**: Yahoo Finance, Alpha Vantage, FRED
- **Benchmark**: S&P 500 (SPY ETF)
- **External**: Federal Reserve (FRED), US Treasury rates

> Raw data files are excluded from version control via `.gitignore`.
