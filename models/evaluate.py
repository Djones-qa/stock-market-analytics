"""
evaluate.py — Forecast metrics, residual analysis, feature importance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


def compute_metrics(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "MAPE": round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
        "R2": round(r2_score(y_true, y_pred), 4),
        "Median_AE": round(np.median(np.abs(y_true - y_pred)), 2),
        "n_samples": len(y_true),
    }


def print_metrics(metrics, model_name="Model"):
    print(f"\n{'='*45}")
    print(f"  {model_name} Forecast Metrics")
    print(f"{'='*45}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"{'='*45}\n")


def plot_residuals(y_true, y_pred, output_dir="visualizations/output"):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(y_true, y_pred, alpha=0.2, s=8, color="#1976D2")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=1.5)
    axes[0].set_title("Predicted vs Actual", fontweight="bold")
    axes[0].set_xlabel("Actual ($)")
    axes[0].set_ylabel("Predicted ($)")
    axes[1].hist(residuals, bins=50, color="#43A047", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title("Residual Distribution", fontweight="bold")
    axes[2].scatter(y_pred, residuals, alpha=0.2, s=8, color="#FF7043")
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_title("Residuals vs Predicted", fontweight="bold")
    plt.tight_layout()
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_dir}/residual_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


def feature_importance_report(pipeline, feature_names, top_n=15, output_dir="visualizations/output"):
    model = pipeline.named_steps.get("model")
    if not hasattr(model, "feature_importances_"):
        print("Model does not support feature_importances_.")
        return pd.DataFrame()
    fi = pd.DataFrame({
        "feature": feature_names[:len(model.feature_importances_)],
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(fi["feature"][::-1], fi["importance"][::-1], color="#1976D2")
    ax.set_title(f"Top {top_n} Feature Importances", fontweight="bold")
    plt.tight_layout()
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_dir}/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    return fi
