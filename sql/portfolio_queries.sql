-- Portfolio current value
SELECT h.ticker, h.shares, h.buy_price, h.current_price,
    ROUND(h.shares * h.buy_price, 2) AS cost_basis,
    ROUND(h.shares * h.current_price, 2) AS current_value,
    ROUND((h.current_price - h.buy_price) / h.buy_price * 100, 2) AS gain_loss_pct
FROM portfolio_holdings h ORDER BY current_value DESC;

-- Portfolio by sector
SELECT h.sector, COUNT(*) AS holdings,
    ROUND(SUM(h.shares * h.buy_price), 2) AS total_cost,
    ROUND(SUM(h.shares * h.current_price), 2) AS total_value,
    ROUND(SUM(h.weight) * 100, 2) AS weight_pct
FROM portfolio_holdings h GROUP BY h.sector ORDER BY total_value DESC;

-- Top/bottom performers
SELECT ticker, ROUND((current_price - buy_price) / buy_price * 100, 2) AS gain_pct
FROM portfolio_holdings ORDER BY gain_pct DESC;

-- Sector diversification check
SELECT sector,
    ROUND(SUM(weight) * 100, 2) AS allocation_pct,
    CASE WHEN SUM(weight) > 0.30 THEN 'OVERWEIGHT'
         WHEN SUM(weight) < 0.05 THEN 'UNDERWEIGHT'
         ELSE 'BALANCED' END AS status
FROM portfolio_holdings GROUP BY sector ORDER BY allocation_pct DESC;

-- Daily portfolio value over time
SELECT sp.date,
    ROUND(SUM(h.shares * sp.close), 2) AS portfolio_value
FROM portfolio_holdings h
JOIN stock_prices sp ON h.ticker = sp.ticker
GROUP BY sp.date ORDER BY sp.date;
