"""
Failure Prediction Machine Learning Module.
Trains, evaluates, and compares classification algorithms (XGBoost, Random Forest,
Gradient Boosting, Logistic Regression) with class imbalance handling, stratified CV,
and ROC/PR evaluation curves.
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from xgboost import XGBClassifier

from src.config import (
    FAILURE_MODEL_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
    TARGET_FAILURE
)
from src.feature_engine import get_feature_matrix
from src.preprocessing import BatteryPreprocessor
from src.models import ModelBundle


def train_and_compare_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: BatteryPreprocessor
) -> Tuple[Dict[str, Dict[str, Any]], str, Any]:
    """
    Benchmarks multiple classification algorithms on the preprocessed training set.
    """
    X_train_scaled = preprocessor.transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    num_neg = int((y_train == 0).sum())
    num_pos = int((y_train == 1).sum())
    scale_pos_weight = num_neg / max(num_pos, 1)

    candidate_models = {
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric=["logloss", "auc"],
            n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            random_state=RANDOM_STATE
        ),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    }

    results = {}
    best_model_name = "XGBoost"
    best_pr_auc = -1.0
    best_estimator = None

    for name, clf in candidate_models.items():
        print(f"  Training {name} classifier...")
        clf.fit(X_train_scaled, y_train)

        # Predictions on holdout test set
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.50).astype(int)

        roc_auc = float(roc_auc_score(y_test, y_pred_proba))
        pr_auc = float(average_precision_score(y_test, y_pred_proba))
        f1 = float(f1_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred))
        acc = float(accuracy_score(y_test, y_pred))
        brier = float(brier_score_loss(y_test, y_pred_proba))

        results[name] = {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "accuracy": round(acc, 4),
            "brier_score": round(brier, 4),
            "estimator": clf
        }

        # Select best model prioritizing PR-AUC (crucial for imbalanced battery failure detection)
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_model_name = name
            best_estimator = clf

    return results, best_model_name, best_estimator


def train_failure_model(
    df: pd.DataFrame,
    output_dir: Optional[str] = None
) -> Tuple[ModelBundle, Dict[str, Any]]:
    """
    Executes the end-to-end Failure Prediction pipeline.
    """
    output_dir = output_dir or str(FAILURE_MODEL_DIR)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("1. PREPARING FEATURE MATRIX FOR FAILURE PREDICTION")
    print("=" * 70)
    X, y, feature_names = get_feature_matrix(df, target_type="failure")
    
    if y is None:
        raise ValueError("Target column 'battery_failure' is missing from dataset.")

    y = y.astype(int)
    print(f"Total samples: {len(X):,}, Total features: {len(feature_names)}")
    print(f"Failure prevalence: {y.mean():.2%} ({y.sum():,} failures / {len(y) - y.sum():,} healthy)")

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Fit Preprocessor exclusively on X_train
    preprocessor = BatteryPreprocessor()
    preprocessor.fit(X_train)

    print("\n" + "=" * 70)
    print("2. BENCHMARKING CLASSIFICATION ALGORITHMS")
    print("=" * 70)
    comparison_results, best_model_name, best_estimator = train_and_compare_classifiers(
        X_train, y_train, X_test, y_test, preprocessor
    )

    print("\nClassifier Benchmark Results:")
    summary_df = pd.DataFrame([
        {
            "Model": k,
            "ROC-AUC": v["roc_auc"],
            "PR-AUC": v["pr_auc"],
            "F1-Score": v["f1_score"],
            "Precision": v["precision"],
            "Recall": v["recall"],
            "Accuracy": v["accuracy"],
            "Brier Score": v["brier_score"]
        }
        for k, v in comparison_results.items()
    ]).sort_values("PR-AUC", ascending=False)
    print(summary_df.to_string(index=False))
    print(f"\nSelected Champion Model: {best_model_name} (PR-AUC: {comparison_results[best_model_name]['pr_auc']:.4f})")

    # Detailed Evaluation of Champion Model
    X_test_scaled = preprocessor.transform(X_test)
    y_test_proba = best_estimator.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = (y_test_proba >= 0.50).astype(int)

    cm = confusion_matrix(y_test, y_test_pred)
    cr = classification_report(y_test, y_test_pred, target_names=["Healthy (0)", "Failure (1)"])
    roc_auc = roc_auc_score(y_test, y_test_proba)
    pr_auc = average_precision_score(y_test, y_test_proba)

    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)

    # Generate and Save Diagnostic Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Healthy (0)', 'Failure (1)'],
                yticklabels=['Healthy (0)', 'Failure (1)'])
    axes[0].set_title(f"Confusion Matrix ({best_model_name})\nPR-AUC: {pr_auc:.3f} | ROC-AUC: {roc_auc:.3f}")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    axes[1].plot(fpr, tpr, color='#1f77b4', lw=2, label=f'{best_model_name} (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    # 3. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    base_rate = float(y.mean())
    axes[2].plot(rec, prec, color='#2ca02c', lw=2, label=f'{best_model_name} (PR-AUC = {pr_auc:.3f})')
    axes[2].axhline(y=base_rate, color='gray', linestyle='--', label=f'Baseline ({base_rate:.2%})')
    axes[2].set_title("Precision-Recall Curve")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].legend(loc="lower left")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "failure_model_evaluation.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # Package into ModelBundle
    bundle_metrics = {
        "champion_model": best_model_name,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1_score": float(f1_score(y_test, y_test_pred)),
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_test_pred)),
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "brier_score": float(brier_score_loss(y_test, y_test_proba)),
        "base_failure_rate": float(base_rate),
        "model_comparison": {k: {m: v[m] for m in v if m != "estimator"} for k, v in comparison_results.items()}
    }

    bundle = ModelBundle(
        model_name="failure_classifier",
        model_type="classification",
        estimator=best_estimator,
        preprocessor=preprocessor,
        feature_names=preprocessor.get_feature_names(),
        metrics=bundle_metrics,
        version="2.0.0",
        metadata={"target": TARGET_FAILURE, "training_samples": len(X_train)}
    )

    bundle.save(output_dir, filename_prefix="failure_model")
    print(f"\nSaved Failure ModelBundle to: {output_dir}")

    return bundle, bundle_metrics
