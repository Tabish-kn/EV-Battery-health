"""
Master Model Training & Artifact Generation Pipeline.
Trains and serializes all 4 intelligence engines:
1. Failure Prediction Classifier (XGBoost champion)
2. State of Health (SOH) Regressor
3. Remaining Useful Life (RUL) Regressor
4. Unsupervised Battery Anomaly Detector
Also updates legacy root bundle for complete backward compatibility.
"""

import os
import joblib
import pandas as pd
from src.data_loader import load_data
from src.failure_prediction import train_failure_model
from src.soh_prediction import train_soh_model
from src.rul_prediction import train_rul_model
from src.anomaly_detection import train_anomaly_model
from src.explainability import BatteryModelExplainer


def main():
    print("=" * 80)
    print("STARTING EV BATTERY INTELLIGENCE PLATFORM MODEL TRAINING SUITE")
    print("=" * 80)

    # Ingest data
    df = load_data()
    print(f"Loaded {len(df):,} telemetry records from dataset.")

    # 1. Train Failure Model
    print("\n>>> [1/4] Training Failure Classification Pipeline...")
    failure_bundle, failure_metrics = train_failure_model(df)

    # 2. Train SOH Model
    print("\n>>> [2/4] Training State of Health (SOH) Regression Pipeline...")
    soh_bundle, soh_metrics = train_soh_model(df)

    # 3. Train RUL Model
    print("\n>>> [3/4] Training Remaining Useful Life (RUL) Regression Pipeline...")
    rul_bundle, rul_metrics = train_rul_model(df)

    # 4. Train Anomaly Detection Model
    print("\n>>> [4/4] Training Unsupervised Anomaly Detection Pipeline...")
    anomaly_bundle, anomaly_metrics = train_anomaly_model(df)

    # 5. Generate SHAP Summary Figure
    print("\n>>> [5/5] Generating Global TreeSHAP Summary Plot...")
    explainer = BatteryModelExplainer(failure_bundle)
    from src.feature_engine import get_feature_matrix
    X_sample, _, _ = get_feature_matrix(df.head(200), target_type="failure")
    explainer.generate_global_summary_plot(X_sample, output_filename="shap_global_summary.png")

    # 6. Legacy Bundle Compatibility
    print("\n>>> Exporting Legacy Pipeline Bundles for Root Compatibility...")
    legacy_bundle = {
        'model': failure_bundle.estimator,
        'scaler': failure_bundle.preprocessor.scaler,
        'num_imputer': failure_bundle.preprocessor.num_imputer,
        'cat_imputer': failure_bundle.preprocessor.cat_imputer,
        'encoder': failure_bundle.preprocessor.encoder,
        'num_cols': failure_bundle.preprocessor.num_cols,
        'cat_cols': failure_bundle.preprocessor.active_cat_cols,
        'all_feature_names': failure_bundle.feature_names,
        'physics_features': [f for f in failure_bundle.feature_names if 'arrhenius' in f or 'stress' in f or 'proxy' in f or 'efc' in f or 'soh' in f or 'strain' in f or 'plating' in f or 'acceleration' in f or 'spread' in f or 'variance' in f or 'rate' in f or 'efficiency' in f or 'density' in f],
        'metrics': {
            'roc_auc': failure_metrics['roc_auc'],
            'pr_auc': failure_metrics['pr_auc'],
            'base_fail_rate': failure_metrics['base_failure_rate']
        }
    }
    joblib.dump(legacy_bundle, "ev_pipeline_bundle.pkl")
    joblib.dump(failure_bundle.estimator, "ev_battery_model.pkl")
    joblib.dump(failure_bundle.preprocessor.scaler, "scaler.pkl")
    joblib.dump(failure_bundle.preprocessor.num_imputer, "imputer.pkl")
    joblib.dump(failure_bundle.feature_names, "feature_columns.pkl")

    print("\n" + "=" * 80)
    print("ALL MODELS TRAINED, EVALUATED, AND SERIALIZED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Failure Model: ROC-AUC = {failure_metrics['roc_auc']:.4f} | PR-AUC = {failure_metrics['pr_auc']:.4f}")
    print(f"SOH Model:     R² = {soh_metrics['r2_score']:.4f} | MAE = {soh_metrics['mae']:.2f}%")
    print(f"RUL Model:     R² = {rul_metrics['r2_score']:.4f} | MAE = {rul_metrics['mae_cycles']:.0f} cycles")
    print(f"Anomaly Model: Anomaly Rate = {anomaly_metrics['anomaly_rate']:.2%}")


if __name__ == "__main__":
    main()
