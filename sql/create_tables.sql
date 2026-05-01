CREATE TABLE IF NOT EXISTS stock_prices (
    price_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    date            DATE NOT NULL,
    open            REAL,
    high            REAL,
    low             REAL,
    close           REAL NOT NULL,
    adj_close       REAL,
    volume          INTEGER DEFAULT 0,
    sector          TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS portfolio_holdings (
    holding_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    shares          REAL NOT NULL,
    buy_price       REAL NOT NULL,
    buy_date        DATE NOT NULL,
    sector          TEXT,
    weight          REAL,
    current_price   REAL,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS technical_indicators (
    indicator_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    date            DATE NOT NULL,
    sma_20          REAL,
    sma_50          REAL,
    sma_200         REAL,
    rsi             REAL,
    macd_line       REAL,
    macd_signal     REAL,
    bb_upper        REAL,
    bb_lower        REAL,
    volatility_20d  REAL,
    atr             REAL,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    forecast_date   DATE NOT NULL,
    model_name      TEXT NOT NULL,
    predicted_price REAL NOT NULL,
    actual_price    REAL,
    error           REAL,
    pct_error       REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON stock_prices(ticker, date);
CREATE INDEX IF NOT EXISTS idx_prices_date ON stock_prices(date);
CREATE INDEX IF NOT EXISTS idx_prices_sector ON stock_prices(sector);
CREATE INDEX IF NOT EXISTS idx_holdings_ticker ON portfolio_holdings(ticker);
CREATE INDEX IF NOT EXISTS idx_indicators_ticker ON technical_indicators(ticker, date);
CREATE INDEX IF NOT EXISTS idx_forecasts_ticker ON forecasts(ticker, forecast_date);
