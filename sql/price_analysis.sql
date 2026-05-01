-- Daily price summary by ticker
SELECT ticker, COUNT(*) AS trading_days,
    ROUND(MIN(close), 2) AS all_time_low,
    ROUND(MAX(close), 2) AS all_time_high,
    ROUND(AVG(close), 2) AS avg_close,
    ROUND(AVG(volume), 0) AS avg_volume
FROM stock_prices GROUP BY ticker ORDER BY avg_close DESC;

-- Monthly performance
SELECT ticker, strftime('%Y-%m', date) AS month,
    ROUND(MIN(close), 2) AS month_low,
    ROUND(MAX(close), 2) AS month_high,
    ROUND(AVG(close), 2) AS avg_close,
    ROUND(SUM(volume), 0) AS total_volume
FROM stock_prices GROUP BY ticker, month ORDER BY ticker, month;

-- Year-over-year returns
WITH yearly AS (
    SELECT ticker, strftime('%Y', date) AS year,
        MIN(close) AS year_open,
        MAX(close) AS year_close
    FROM stock_prices GROUP BY ticker, year
)
SELECT curr.ticker, curr.year,
    ROUND(curr.year_close, 2) AS close_price,
    ROUND((curr.year_close - prev.year_close) / prev.year_close * 100, 2) AS yoy_return_pct
FROM yearly curr LEFT JOIN yearly prev
    ON curr.ticker = prev.ticker
    AND CAST(curr.year AS INT) = CAST(prev.year AS INT) + 1
ORDER BY curr.ticker, curr.year;

-- 52-week high/low
SELECT ticker,
    ROUND(MIN(low), 2) AS week52_low,
    ROUND(MAX(high), 2) AS week52_high,
    ROUND((MAX(high) - MIN(low)) / MIN(low) * 100, 2) AS range_pct
FROM stock_prices
WHERE date >= DATE('now', '-365 days')
GROUP BY ticker ORDER BY range_pct DESC;

-- Volume spike detection
SELECT ticker, date, volume,
    ROUND(volume * 1.0 / AVG(volume) OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING), 2) AS volume_ratio
FROM stock_prices
ORDER BY volume_ratio DESC LIMIT 50;
