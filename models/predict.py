"""
predict.py — Load saved model and generate price forecasts.
"""

import pandas as pd
import numpy as np
import joblib
from glob import glob
from src.utils import format_currency


def load_latest_model(model_dir="models/saved_models"):
    model_files = sorted(glob(f"{model_dir}/*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No saved models in {model_dir}/")
    latest = model_files[-1]
    pipeline = joblib.load(latest)
    print(f"Loaded: {latest}")
    return pipeline, latest


def predict_price(pipeline, features_dict):
    df = pd.DataFrame([features_dict])
    prediction = pipeline.predict(df)[0]
    return {"predicted_price": round(prediction, 2), "formatted": format_currency(prediction)}


def predict_batch(pipeline, df, feature_cols, output_col="predicted_price"):
    df = df.copy()
    X = df[feature_cols].fillna(method="ffill")
    df[output_col] = np.round(pipeline.predict(X), 2)
    print(f"Generated {len(df):,} predictions")
    return df


def forecast_accuracy(df, actual_col="close", predicted_col="predicted_price"):
    df = df.copy()
    df["error"] = df[predicted_col] - df[actual_col]
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = (df["error"] / df[actual_col] * 100).round(2)
    print(f"MAE: {format_currency(df['abs_error'].mean())}")
    print(f"MAPE: {df['pct_error'].abs().mean():.2f}%")
    return df
