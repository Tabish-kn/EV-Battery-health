"""
Unsupervised Anomaly Detection Module for EV Battery Telemetry.
Uses Isolation Forest to identify multivariate sensor drifts, abnormal thermal spikes,
voltage imbalances, and degradation anomalies without reliance on ground-truth failure labels.
"""

import os
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest

from src.config import (
    ANOMALY_MODEL_DIR,
    FIGURES_DIR,
    RANDOM_STATE
)
from src.feature_engine import get_feature_matrix
from src.preprocessing import BatteryPreprocessor
from src.models import ModelBundle


class BatteryAnomalyDetector:
    """
    Wrapper for Isolation Forest with calibrated anomaly scores (0-100),
    severity categorization, and sensor feature attribution.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = RANDOM_STATE):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=150,
            n_jobs=-1
        )
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.score_offset: float = 0.0

    def fit(self, X: np.ndarray):
        self.model.fit(X)
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0)
        self.feature_stds[self.feature_stds == 0] = 1.0

        # Calibration offset: isolation forest score_samples is negative (closer to 0 or positive is normal)
        raw_scores = self.model.score_samples(X)
        self.score_min = float(raw_scores.min())
        self.score_max = float(raw_scores.max())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 for normal, -1 for anomaly."""
        return self.model.predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Returns raw decision score."""
        return self.model.score_samples(X)

    def calculate_anomaly_score_0_100(self, X: np.ndarray) -> np.ndarray:
        """
        Converts Isolation Forest score into a 0 to 100 anomaly severity index
        (0 = completely nominal, 100 = severe multivariate outlier).
        """
        raw_scores = self.model.score_samples(X)
        # Invert: lower raw_score means more anomalous
        normalized = (self.score_max - raw_scores) / max(self.score_max - self.score_min, 1e-6)
        scores_100 = np.clip(normalized * 100.0, 0.0, 100.0)
        return scores_100

    def attribute_anomalous_features(
        self,
        X_sample_scaled: np.ndarray,
        feature_names: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Calculates normalized Z-score deviations from the nominal distribution
        to identify which sensors drive the anomaly.
        """
        if self.feature_means is None or self.feature_stds is None:
            return []

        z_scores = np.abs((X_sample_scaled - self.feature_means) / self.feature_stds)
        if z_scores.ndim > 1:
            z_scores = z_scores[0]

        top_indices = np.argsort(z_scores)[::-1][:top_k]
        attributions = []
        for idx in top_indices:
            attributions.append({
                "feature": feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
                "z_score": round(float(z_scores[idx]), 2),
                "severity": "CRITICAL" if z_scores[idx] > 3.0 else ("ELEVATED" if z_scores[idx] > 2.0 else "NOMINAL")
            })
        return attributions


def train_anomaly_model(
    df: pd.DataFrame,
    output_dir: Optional[str] = None
) -> Tuple[ModelBundle, Dict[str, Any]]:
    """
    Trains and packages the Unsupervised Battery Anomaly Detection Engine.
    """
    output_dir = output_dir or str(ANOMALY_MODEL_DIR)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("1. PREPARING FEATURE MATRIX FOR UNSUPERVISED ANOMALY DETECTION")
    print("=" * 70)
    X, _, feature_names = get_feature_matrix(df, target_type="anomaly")
    print(f"Ingested {len(X):,} samples across {len(feature_names)} features.")

    preprocessor = BatteryPreprocessor()
    X_scaled = preprocessor.fit_transform(X)

    detector = BatteryAnomalyDetector(contamination=0.05, random_state=RANDOM_STATE)
    detector.fit(X_scaled)

    raw_scores = detector.score_samples(X_scaled)
    anomaly_scores_100 = detector.calculate_anomaly_score_0_100(X_scaled)
    preds = detector.predict(X_scaled)  # -1 anomaly, 1 normal
    anomaly_flags = (preds == -1).astype(int)

    n_anomalies = int(anomaly_flags.sum())
    print(f"\nDetected {n_anomalies:,} anomalies ({n_anomalies / len(df):.2%} contamination rate)")
    print(f"Anomaly Score Range: [{anomaly_scores_100.min():.1f}, {anomaly_scores_100.max():.1f}] / 100")

    # Diagnostic Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Anomaly Score Distribution
    sns.histplot(anomaly_scores_100, bins=50, kde=True, ax=axes[0], color='#d97706')
    axes[0].set_title(f"Unsupervised Battery Anomaly Score Distribution\n(Mean: {anomaly_scores_100.mean():.1f}/100)")
    axes[0].set_xlabel("Anomaly Severity Score (0 - 100)")
    axes[0].set_ylabel("Battery Pack Count")
    axes[0].grid(True, alpha=0.3)

    # 2. Anomaly vs SOH / Failure
    if 'state_of_health' in df.columns:
        axes[1].scatter(
            df['state_of_health'],
            anomaly_scores_100,
            c=anomaly_flags,
            cmap='coolwarm',
            alpha=0.4,
            s=15
        )
        axes[1].set_title("Anomaly Score vs State of Health (SOH %)")
        axes[1].set_xlabel("State of Health (%)")
        axes[1].set_ylabel("Anomaly Severity Score")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "anomaly_model_evaluation.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    bundle_metrics = {
        "algorithm": "IsolationForest",
        "contamination": 0.05,
        "anomalies_detected": n_anomalies,
        "anomaly_rate": float(n_anomalies / len(df)),
        "mean_anomaly_score": float(anomaly_scores_100.mean()),
        "max_anomaly_score": float(anomaly_scores_100.max())
    }

    bundle = ModelBundle(
        model_name="anomaly_detector",
        model_type="anomaly",
        estimator=detector,
        preprocessor=preprocessor,
        feature_names=preprocessor.get_feature_names(),
        metrics=bundle_metrics,
        version="2.0.0",
        metadata={"algorithm": "IsolationForest", "samples": len(X)}
    )

    bundle.save(output_dir, filename_prefix="anomaly_model")
    print(f"\nSaved Anomaly ModelBundle to: {output_dir}")

    return bundle, bundle_metrics
