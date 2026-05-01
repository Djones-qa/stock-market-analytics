"""
technical_signals.py — Generate buy/sell signals from technical indicators.
"""


def ma_crossover_signals(df):
    """Generate signals from SMA 50/200 crossover."""
    if "sma_50" not in df.columns or "sma_200" not in df.columns:
        return df
    df = df.copy()
    df["ma_signal"] = 0
    df.loc[(df["sma_50"] > df["sma_200"])
           & (df["sma_50"].shift(1) <= df["sma_200"].shift(1)), "ma_signal"] = 1
    df.loc[(df["sma_50"] < df["sma_200"])
           & (df["sma_50"].shift(1) >= df["sma_200"].shift(1)), "ma_signal"] = -1
    return df


def rsi_signals(df, overbought=70, oversold=30):
    """Generate signals from RSI levels."""
    if "rsi" not in df.columns:
        return df
    df = df.copy()
    df["rsi_signal"] = 0
    df.loc[df["rsi"] < oversold, "rsi_signal"] = 1
    df.loc[df["rsi"] > overbought, "rsi_signal"] = -1
    return df


def macd_signals(df):
    """Generate signals from MACD crossover."""
    if "macd_line" not in df.columns or "macd_signal" not in df.columns:
        return df
    df = df.copy()
    df["macd_trade_signal"] = 0
    df.loc[(df["macd_line"] > df["macd_signal"])
           & (df["macd_line"].shift(1) <= df["macd_signal"].shift(1)), "macd_trade_signal"] = 1
    df.loc[(df["macd_line"] < df["macd_signal"])
           & (df["macd_line"].shift(1) >= df["macd_signal"].shift(1)), "macd_trade_signal"] = -1
    return df


def bollinger_signals(df):
    """Generate signals from Bollinger Band position."""
    if "bb_position" not in df.columns:
        return df
    df = df.copy()
    df["bb_signal"] = 0
    df.loc[df["bb_position"] < 0, "bb_signal"] = 1
    df.loc[df["bb_position"] > 1, "bb_signal"] = -1
    return df


def composite_signal(df):
    """Combine all signals into a composite score."""
    df = df.copy()
    signal_cols = [c for c in ["ma_signal", "rsi_signal", "macd_trade_signal", "bb_signal"] if c in df.columns]
    if signal_cols:
        df["composite_signal"] = df[signal_cols].sum(axis=1)
        df["signal_strength"] = df["composite_signal"].abs()
    return df


def generate_all_signals(df):
    """Apply all technical signal generators."""
    print("Generating trading signals...")
    df = ma_crossover_signals(df)
    df = rsi_signals(df)
    df = macd_signals(df)
    df = bollinger_signals(df)
    df = composite_signal(df)
    signal_cols = [c for c in df.columns if "signal" in c]
    print(f"  Generated {len(signal_cols)} signal columns")
    return df
