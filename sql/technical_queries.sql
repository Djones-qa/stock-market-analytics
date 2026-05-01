-- Current technical indicator snapshot
SELECT t.ticker, t.date, t.rsi,
    CASE WHEN t.rsi > 70 THEN 'OVERBOUGHT'
         WHEN t.rsi < 30 THEN 'OVERSOLD'
         ELSE 'NEUTRAL' END AS rsi_status,
    t.sma_50, t.sma_200,
    CASE WHEN t.sma_50 > t.sma_200 THEN 'BULLISH' ELSE 'BEARISH' END AS ma_trend,
    t.volatility_20d
FROM technical_indicators t
WHERE t.date = (SELECT MAX(date) FROM technical_indicators WHERE ticker = t.ticker)
ORDER BY t.ticker;

-- Golden cross events
SELECT ticker, date, sma_50, sma_200
FROM technical_indicators
WHERE sma_50 > sma_200
  AND ticker || date IN (
    SELECT ticker || date FROM technical_indicators
    WHERE sma_50 <= sma_200
  )
ORDER BY date DESC LIMIT 20;

-- Highest volatility stocks
SELECT ticker,
    ROUND(AVG(volatility_20d), 4) AS avg_vol,
    ROUND(MAX(volatility_20d), 4) AS max_vol,
    ROUND(AVG(atr), 2) AS avg_atr
FROM technical_indicators
WHERE date >= DATE('now', '-90 days')
GROUP BY ticker ORDER BY avg_vol DESC;

-- RSI extremes in last 30 days
SELECT ticker, date, rsi,
    CASE WHEN rsi > 70 THEN 'OVERBOUGHT' ELSE 'OVERSOLD' END AS condition
FROM technical_indicators
WHERE date >= DATE('now', '-30 days')
  AND (rsi > 70 OR rsi < 30)
ORDER BY date DESC;
