"""
Remaining Useful Life (RUL) Prediction Module.
Estimates remaining useful battery cycles until the end-of-life degradation threshold
(typically 70-80% SOH retention), provides prediction intervals, and benchmarks regressors.
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
    RUL_MODEL_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
    TARGET_RUL
)
from src.feature_engine import get_feature_matrix
from src.preprocessing import BatteryPreprocessor
from src.models import ModelBundle


class RULPredictorWithIntervals:
    """
    RUL Predictor wrapping a base point regressor and computing estimated prediction intervals
    based on calibrated empirical residual standard errors.
    """

    def __init__(self, base_estimator: Any, residual_std: float = 250.0):
        self.base_estimator = base_estimator
        self.residual_std = residual_std

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.base_estimator.fit(X, y)
        preds = self.base_estimator.predict(X)
        residuals = y - preds
        self.residual_std = float(np.std(residuals))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, self.base_estimator.predict(X))

    def predict_with_intervals(
        self,
        X: np.ndarray,
        confidence: float = 0.90
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (predicted_rul, lower_bound, upper_bound) at given confidence level.
        """
        point_pred = self.predict(X)
        z_multiplier = 1.645 if confidence >= 0.90 else 1.96
        margin = z_multiplier * self.residual_std
        
        lower_bound = np.maximum(0.0, point_pred - margin)
        upper_bound = point_pred + margin
        return point_pred, lower_bound, upper_bound


def train_and_compare_rul_regressors(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: BatteryPreprocessor
) -> Tuple[Dict[str, Dict[str, Any]], str, Any]:
    """
    Benchmarks algorithms for RUL (Remaining Useful Cycles) estimation.
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
        print(f"  Training {name} for RUL cycle prediction...")
        reg.fit(X_train_scaled, y_train)

        y_pred = np.maximum(0.0, reg.predict(X_test_scaled))

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(mean_absolute_percentage_error(y_test, y_pred) * 100.0)

        results[name] = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "mape_percent": round(mape, 2),
            "estimator": reg
        }

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_estimator = reg

    return results, best_model_name, best_estimator


def train_rul_model(
    df: pd.DataFrame,
    output_dir: Optional[str] = None
) -> Tuple[ModelBundle, Dict[str, Any]]:
    """
    Executes end-to-end Remaining Useful Life (RUL) regression pipeline.
    """
    output_dir = output_dir or str(RUL_MODEL_DIR)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("1. PREPARING FEATURE MATRIX FOR RUL PREDICTION")
    print("=" * 70)
    X, y, feature_names = get_feature_matrix(df, target_type="rul")

    if y is None:
        raise ValueError(f"RUL target column '{TARGET_RUL}' is missing from dataset.")

    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    print(f"Valid RUL training samples: {len(X):,}, RUL Cycle Range: [{y.min():,.0f}, {y.max():,.0f} cycles]")
    print(f"RUL Mean: {y.mean():,.0f} cycles, RUL Median: {y.median():,.0f} cycles")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    preprocessor = BatteryPreprocessor()
    preprocessor.fit(X_train)

    print("\n" + "=" * 70)
    print("2. BENCHMARKING RUL REGRESSION ALGORITHMS")
    print("=" * 70)
    comparison_results, best_model_name, best_estimator = train_and_compare_rul_regressors(
        X_train, y_train, X_test, y_test, preprocessor
    )

    summary_df = pd.DataFrame([
        {
            "Model": k,
            "R² Score": v["r2"],
            "MAE (Cycles)": v["mae"],
            "RMSE (Cycles)": v["rmse"],
            "MAPE (%)": v["mape_percent"]
        }
        for k, v in comparison_results.items()
    ]).sort_values("R² Score", ascending=False)
    print("\nRUL Regressor Benchmark Results:")
    print(summary_df.to_string(index=False))
    print(f"\nSelected Champion RUL Model: {best_model_name} (R² = {comparison_results[best_model_name]['r2']:.4f})")

    # Fit wrapped interval predictor
    X_train_scaled = preprocessor.transform(X_train)
    wrapped_predictor = RULPredictorWithIntervals(best_estimator)
    wrapped_predictor.fit(X_train_scaled, y_train.values)

    X_test_scaled = preprocessor.transform(X_test)
    y_test_pred, y_lower, y_upper = wrapped_predictor.predict_with_intervals(X_test_scaled)

    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred)
    mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100.0

    # Diagnostic Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Actual vs Predicted RUL
    axes[0].scatter(y_test, y_test_pred, alpha=0.3, color='#059669', edgecolors='none', s=20)
    min_val, max_val = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], color='#ef4444', linestyle='--', lw=2, label='Ideal 1:1')
    axes[0].set_title(f"Actual vs Predicted RUL\nR²: {r2:.3f} | MAE: {mae:.0f} cycles")
    axes[0].set_xlabel("Actual Remaining Useful Cycles")
    axes[0].set_ylabel("Predicted Remaining Useful Cycles")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Residual Distribution
    residuals = y_test - y_test_pred
    sns.histplot(residuals, kde=True, ax=axes[1], color='#f59e0b', bins=40)
    axes[1].axvline(0, color='#ef4444', linestyle='--')
    axes[1].set_title(f"RUL Residual Distribution\nMean: {residuals.mean():.1f} | Std: {residuals.std():.1f} cycles")
    axes[1].set_xlabel("Residual (Actual - Predicted Cycles)")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, alpha=0.3)

    # 3. RUL vs Cycle Count
    if 'cycle_count' in X_test.columns:
        axes[2].scatter(X_test['cycle_count'], y_test_pred, alpha=0.3, color='#6366f1', s=20, label='Predicted RUL')
        axes[2].set_title("Predicted RUL vs Completed Cycle Count")
        axes[2].set_xlabel("Completed Cycle Count")
        axes[2].set_ylabel("Predicted Remaining Useful Cycles")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "rul_model_evaluation.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    bundle_metrics = {
        "champion_model": best_model_name,
        "r2_score": float(r2),
        "mae_cycles": float(mae),
        "rmse_cycles": float(rmse),
        "mape_percent": float(mape),
        "residual_std": float(wrapped_predictor.residual_std),
        "model_comparison": {k: {m: v[m] for m in v if m != "estimator"} for k, v in comparison_results.items()}
    }

    bundle = ModelBundle(
        model_name="rul_regressor",
        model_type="regression",
        estimator=wrapped_predictor,
        preprocessor=preprocessor,
        feature_names=preprocessor.get_feature_names(),
        metrics=bundle_metrics,
        version="2.0.0",
        metadata={"target": TARGET_RUL, "training_samples": len(X_train)}
    )

    bundle.save(output_dir, filename_prefix="rul_model")
    print(f"\nSaved RUL ModelBundle to: {output_dir}")

    return bundle, bundle_metrics
