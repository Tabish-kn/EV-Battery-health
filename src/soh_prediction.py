"""
State of Health (SOH) Estimation & Degradation Regression Module.
Trains, evaluates, and benchmarks regression algorithms (XGBoost, Random Forest,
Gradient Boosting, Ridge) to predict battery State of Health percentage (%)
while strictly preventing target leakage.
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor

from src.config import (
    SOH_MODEL_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
    TARGET_SOH,
    TARGET_SOH_ALT
)
from src.feature_engine import get_feature_matrix
from src.preprocessing import BatteryPreprocessor
from src.models import ModelBundle


def train_and_compare_soh_regressors(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: BatteryPreprocessor
) -> Tuple[Dict[str, Dict[str, Any]], str, Any]:
    """
    Benchmarks regression models for State of Health (SOH %) estimation.
    """
    X_train_scaled = preprocessor.transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    candidate_models = {
        "XGBoostRegressor": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            random_state=RANDOM_STATE
        ),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE)
    }

    results = {}
    best_model_name = "XGBoostRegressor"
    best_r2 = -float("inf")
    best_estimator = None

    for name, reg in candidate_models.items():
        print(f"  Training {name} for SOH estimation...")
        reg.fit(X_train_scaled, y_train)

        y_pred = reg.predict(X_test_scaled)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(mean_absolute_percentage_error(y_test, y_pred) * 100.0)

        results[name] = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "mape_percent": round(mape, 4),
            "estimator": reg
        }

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_estimator = reg

    return results, best_model_name, best_estimator


def train_soh_model(
    df: pd.DataFrame,
    output_dir: Optional[str] = None
) -> Tuple[ModelBundle, Dict[str, Any]]:
    """
    Executes end-to-end State of Health (SOH) regression pipeline.
    """
    output_dir = output_dir or str(SOH_MODEL_DIR)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("1. PREPARING FEATURE MATRIX FOR SOH ESTIMATION")
    print("=" * 70)
    X, y, feature_names = get_feature_matrix(df, target_type="soh")

    if y is None:
        raise ValueError(f"SOH target column '{TARGET_SOH}' is missing from dataset.")

    # Filter any records where SOH is NaN
    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    print(f"Valid SOH training samples: {len(X):,}, SOH Range: [{y.min():.2f}%, {y.max():.2f}%]")
    print(f"SOH Mean: {y.mean():.2f}%, SOH Median: {y.median():.2f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    preprocessor = BatteryPreprocessor()
    preprocessor.fit(X_train)

    print("\n" + "=" * 70)
    print("2. BENCHMARKING SOH REGRESSION ALGORITHMS")
    print("=" * 70)
    comparison_results, best_model_name, best_estimator = train_and_compare_soh_regressors(
        X_train, y_train, X_test, y_test, preprocessor
    )

    summary_df = pd.DataFrame([
        {
            "Model": k,
            "R² Score": v["r2"],
            "MAE (% SOH)": v["mae"],
            "RMSE (% SOH)": v["rmse"],
            "MAPE (%)": v["mape_percent"]
        }
        for k, v in comparison_results.items()
    ]).sort_values("R² Score", ascending=False)
    print("\nSOH Regressor Benchmark Results:")
    print(summary_df.to_string(index=False))
    print(f"\nSelected Champion SOH Model: {best_model_name} (R² = {comparison_results[best_model_name]['r2']:.4f})")

    # Evaluation of Champion Model
    X_test_scaled = preprocessor.transform(X_test)
    y_test_pred = best_estimator.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred)
    mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100.0

    # Diagnostic Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Actual vs Predicted SOH
    axes[0].scatter(y_test, y_test_pred, alpha=0.3, color='#2563eb', edgecolors='none', s=20)
    min_val, max_val = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], color='#ef4444', linestyle='--', lw=2, label='Ideal 1:1')
    axes[0].set_title(f"Actual vs Predicted SOH\nR²: {r2:.3f} | MAE: {mae:.2f}%")
    axes[0].set_xlabel("Actual State of Health (%)")
    axes[0].set_ylabel("Predicted State of Health (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Residual Distribution
    residuals = y_test - y_test_pred
    sns.histplot(residuals, kde=True, ax=axes[1], color='#10b981', bins=40)
    axes[1].axvline(0, color='#ef4444', linestyle='--')
    axes[1].set_title(f"SOH Prediction Residual Distribution\nMean: {residuals.mean():.3f}% | Std: {residuals.std():.3f}%")
    axes[1].set_xlabel("Residual (Actual - Predicted %)")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, alpha=0.3)

    # 3. Residuals vs Actual SOH (Heteroscedasticity Check)
    axes[2].scatter(y_test, residuals, alpha=0.3, color='#8b5cf6', s=20)
    axes[2].axhline(0, color='#ef4444', linestyle='--')
    axes[2].set_title("Residuals vs Actual SOH")
    axes[2].set_xlabel("Actual State of Health (%)")
    axes[2].set_ylabel("Residual (%)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "soh_model_evaluation.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    bundle_metrics = {
        "champion_model": best_model_name,
        "r2_score": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape_percent": float(mape),
        "model_comparison": {k: {m: v[m] for m in v if m != "estimator"} for k, v in comparison_results.items()}
    }

    bundle = ModelBundle(
        model_name="soh_regressor",
        model_type="regression",
        estimator=best_estimator,
        preprocessor=preprocessor,
        feature_names=preprocessor.get_feature_names(),
        metrics=bundle_metrics,
        version="2.0.0",
        metadata={"target": TARGET_SOH, "training_samples": len(X_train)}
    )

    bundle.save(output_dir, filename_prefix="soh_model")
    print(f"\nSaved SOH ModelBundle to: {output_dir}")

    return bundle, bundle_metrics
