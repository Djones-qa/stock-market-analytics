"""
train.py — Train price forecasting models with time-series cross-validation.
"""

import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import load_config  # noqa: E402


def prepare_features(df, config=None):
    if config is None:
        config = load_config()
    feature_cols = [
        "month", "day_of_week", "quarter",
        "sma_20", "sma_50", "ema_20",
        "rsi", "macd_line", "macd_histogram",
        "bb_position", "bb_width",
        "volatility_20d", "atr",
        "volume_ratio", "daily_return",
        "return_5d", "return_20d",
        "price_lag_1", "price_lag_5", "price_lag_20",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    target = config["data"]["price_column"]
    df_clean = df[feature_cols + [target]].dropna()
    X = df_clean[feature_cols]
    y = df_clean[target]
    return X, y, feature_cols


def get_models():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }


def train_and_compare(df, config=None, save_best=True):
    if config is None:
        config = load_config()
    X, y, feature_cols = prepare_features(df, config)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    tscv = TimeSeriesSplit(n_splits=5)
    models = get_models()
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0
        )
    except ImportError:
        pass
    results = []
    best_mae, best_name, best_pipe = float("inf"), None, None
    for name, model in models.items():
        print(f"Training {name}...", end=" ")
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        cv = cross_val_score(pipe, X_train, y_train, cv=tscv,
                             scoring="neg_mean_absolute_error", n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results.append({
            "model": name, "cv_mae": round(-cv.mean(), 2),
            "test_mae": round(mae, 2), "test_rmse": round(rmse, 2),
            "test_r2": round(r2, 4)
        })
        print(f"MAE=${mae:.2f} | R2={r2:.4f}")
        if mae < best_mae:
            best_mae, best_name, best_pipe = mae, name, pipe
    results_df = pd.DataFrame(results).sort_values("test_mae")
    print(f"\nBest: {best_name} (MAE=${best_mae:.2f})")
    if save_best and best_pipe:
        save_dir = "models/saved_models"
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{save_dir}/{best_name}_{ts}.joblib"
        joblib.dump(best_pipe, path)
        print(f"Saved to {path}")
    return results_df, best_pipe, (X_test, y_test)
