"""
End-to-End ML Pipeline for EV Battery Health & Failure Prediction.
Performs EDA summary, sensor spike clipping, physics feature engineering,
imputation, one-hot encoding, scaling, XGBoost training with scale_pos_weight,
comprehensive evaluation, and artifact serialization.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from xgboost import XGBClassifier

from feature_engine import clip_sensor_spikes, engineer_physics_features


def run_pipeline(csv_path: str = "ev_battery_health.csv", output_dir: str = "."):
    print("=" * 70)
    print("1. LOADING & PREVIEWING EV BATTERY TELEMETRY DATA")
    print("=" * 70)
    df_raw = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")
    print(f"Target distribution ('battery_failure'):")
    print(df_raw['battery_failure'].value_counts(normalize=False))
    base_fail_rate = df_raw['battery_failure'].mean()
    print(f"Base failure rate: {base_fail_rate:.2%}")

    print("\n" + "=" * 70)
    print("2. SENSOR SPIKE CLIPPING & PHYSICS FEATURE ENGINEERING")
    print("=" * 70)
    df_clean = clip_sensor_spikes(df_raw)
    df_feat = engineer_physics_features(df_clean)
    print(f"Shape after feature engineering: {df_feat.shape}")

    # Verify physics features presence
    physics_features = [
        'arrhenius_aging_factor',
        'cell_temperature_variance_delta',
        'cell_voltage_spread',
        'fast_charge_cumulative_stress',
        'resistance_growth_proxy',
        'efc_per_1000km',
        'composite_physical_stress',
        'soh_wear_loss_per_cycle',
        'thermal_dissipation_strain',
        'ir_per_kwh',
        'lithium_plating_risk',
        'degradation_acceleration_index',
    ]
    print("\nCalculated Physics Features Summary (Mean +/- Std):")
    for f in physics_features:
        print(f" - {f:<35}: {df_feat[f].mean():.4f} +/- {df_feat[f].std():.4f}")

    print("\n" + "=" * 70)
    print("3. FEATURE SELECTION & PREPROCESSING")
    print("=" * 70)
    # Target
    y = df_feat['battery_failure'].astype(int)

    # Columns to drop (identifiers / high-cardinality metadata)
    drop_cols = ['vehicle_id', 'battery_serial', 'battery_failure']
    X_raw = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])

    # Categorical columns to encode
    potential_cat_cols = [
        'battery_chemistry', 'drive_type', 'fleet_or_private', 'terrain_type',
        'vehicle_type', 'battery_status'
    ]
    cat_cols = [c for c in potential_cat_cols if c in X_raw.columns]
    num_cols = [c for c in X_raw.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(X_raw[c])]

    print(f"Numeric features count: {len(num_cols)}")
    print(f"Categorical features to encode: {cat_cols}")

    # Split: 80% train, 20% test stratified
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train set: {X_train_raw.shape[0]:,} samples (failures: {y_train.sum()})")
    print(f"Test set:  {X_test_raw.shape[0]:,} samples (failures: {y_test.sum()})")

    # Numeric Imputer (Median)
    num_imputer = SimpleImputer(strategy='median')
    X_train_num_imp = num_imputer.fit_transform(X_train_raw[num_cols])
    X_test_num_imp = num_imputer.transform(X_test_raw[num_cols])

    # Categorical Imputer & OneHotEncoder
    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train_cat_imp = cat_imputer.fit_transform(X_train_raw[cat_cols])
        X_test_cat_imp = cat_imputer.transform(X_test_raw[cat_cols])

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train_cat_encoded = encoder.fit_transform(X_train_cat_imp)
        X_test_cat_encoded = encoder.transform(X_test_cat_imp)
        cat_feature_names = encoder.get_feature_names_out(cat_cols).tolist()
    else:
        cat_imputer = None
        encoder = None
        X_train_cat_encoded = np.empty((len(X_train_raw), 0))
        X_test_cat_encoded = np.empty((len(X_test_raw), 0))
        cat_feature_names = []

    # Combine processed arrays
    all_feature_names = num_cols + cat_feature_names
    X_train_combined = np.hstack([X_train_num_imp, X_train_cat_encoded])
    X_test_combined = np.hstack([X_test_num_imp, X_test_cat_encoded])

    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_combined)

    print(f"Total features after encoding & scaling: {len(all_feature_names)}")

    print("\n" + "=" * 70)
    print("4. TRAINING XGBOOST CLASSIFIER WITH CLASS-IMBALANCE HANDLING")
    print("=" * 70)
    num_negative = (y_train == 0).sum()
    num_positive = (y_train == 1).sum()
    scale_pos_weight = num_negative / max(num_positive, 1)
    print(f"Class count: Healthy (0) = {num_negative}, Failure (1) = {num_positive}")
    print(f"Calculated scale_pos_weight = {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric=['logloss', 'auc'],
        early_stopping_rounds=30,
        n_jobs=-1
    )

    eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
    model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=False)
    print(f"Best iteration: {model.best_iteration}")

    print("\n" + "=" * 70)
    print("5. MODEL EVALUATION ON 20% TEST SPLIT")
    print("=" * 70)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = (y_test_proba >= 0.50).astype(int)

    roc_auc = roc_auc_score(y_test, y_test_proba)
    pr_auc = average_precision_score(y_test, y_test_proba)
    cm = confusion_matrix(y_test, y_test_pred)
    cr = classification_report(y_test, y_test_pred, target_names=['Healthy (0)', 'Failure (1)'])

    print(f"Test ROC-AUC Score: {roc_auc:.4f}")
    print(f"Test PR-AUC Score:  {pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)

    # Plot & Save Evaluation Figures
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Healthy (0)', 'Failure (1)'],
                yticklabels=['Healthy (0)', 'Failure (1)'])
    axes[0].set_title(f"Confusion Matrix (Test Split)\nROC-AUC: {roc_auc:.3f}")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    axes[1].plot(fpr, tpr, color='#1f77b4', lw=2, label=f'XGBoost (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    # 3. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    axes[2].plot(rec, prec, color='#2ca02c', lw=2, label=f'XGBoost (PR-AUC = {pr_auc:.3f})')
    axes[2].axhline(y=base_fail_rate, color='gray', linestyle='--', label=f'Baseline ({base_fail_rate:.2%})')
    axes[2].set_title("Precision-Recall Curve")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].legend(loc="lower left")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    eval_fig_path = os.path.join(output_dir, "model_evaluation_metrics.png")
    plt.savefig(eval_fig_path, dpi=300)
    plt.close()
    print(f"Saved evaluation plots to: {eval_fig_path}")

    # Top Feature Importance
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print("\nTop 15 Most Important Features:")
    for idx, row in fi_df.head(15).iterrows():
        is_physics = row['Feature'] in physics_features
        tag = " [PHYSICS ENGINE]" if is_physics else ""
        print(f" {row['Feature']:<35}: {row['Importance']:.4f}{tag}")

    # Plot top feature importances
    plt.figure(figsize=(10, 8))
    top_fi = fi_df.head(20).copy()
    top_fi['Color'] = top_fi['Feature'].apply(lambda x: '#e74c3c' if x in physics_features else '#3498db')
    plt.barh(top_fi['Feature'][::-1], top_fi['Importance'][::-1], color=top_fi['Color'][::-1])
    plt.title("Top 20 Features - EV Battery Failure Prediction\n(Red: Engineered Physics Metrics, Blue: Raw Sensors)")
    plt.xlabel("XGBoost Relative Importance")
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fi_fig_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(fi_fig_path, dpi=300)
    plt.close()
    print(f"Saved feature importance plot to: {fi_fig_path}")

    print("\n" + "=" * 70)
    print("6. EXPORTING PIPELINE ARTIFACTS")
    print("=" * 70)
    model_path = os.path.join(output_dir, "ev_battery_model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    imputer_path = os.path.join(output_dir, "imputer.pkl")
    features_path = os.path.join(output_dir, "feature_columns.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(num_imputer, imputer_path)
    joblib.dump(all_feature_names, features_path)

    # Complete pipeline bundle for Streamlit inference
    pipeline_bundle = {
        'model': model,
        'scaler': scaler,
        'num_imputer': num_imputer,
        'cat_imputer': cat_imputer,
        'encoder': encoder,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'all_feature_names': all_feature_names,
        'physics_features': physics_features,
        'metrics': {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'base_fail_rate': float(base_fail_rate)
        }
    }
    bundle_path = os.path.join(output_dir, "ev_pipeline_bundle.pkl")
    joblib.dump(pipeline_bundle, bundle_path)

    print(f"Exported model:           {model_path}")
    print(f"Exported scaler:          {scaler_path}")
    print(f"Exported imputer:         {imputer_path}")
    print(f"Exported feature columns: {features_path}")
    print(f"Exported pipeline bundle: {bundle_path}")
    print("\nPipeline execution complete successfully!")

    return pipeline_bundle


if __name__ == "__main__":
    run_pipeline()
