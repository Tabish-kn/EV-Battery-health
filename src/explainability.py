"""
Explainable AI (XAI) & SHAP Interpretability Engine.
Implements TreeSHAP explainers for failure classification, global attribution plots,
and local waterfall/force bar attributions for individual battery diagnostic telemetry.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.models import ModelBundle
from src.config import FIGURES_DIR


class BatteryModelExplainer:
    """
    SHAP explainability engine for battery health diagnostic models.
    """

    def __init__(self, model_bundle: ModelBundle):
        self.bundle = model_bundle
        self.estimator = model_bundle.estimator
        self.preprocessor = model_bundle.preprocessor
        self.feature_names = model_bundle.feature_names
        self.explainer = None

        if SHAP_AVAILABLE:
            try:
                # TreeExplainer for XGBoost / RandomForest
                self.explainer = shap.TreeExplainer(self.estimator)
            except Exception:
                try:
                    self.explainer = shap.Explainer(self.estimator)
                except Exception:
                    self.explainer = None

    def explain_instance(
        self,
        X_single: pd.DataFrame,
        top_k: int = 8
    ) -> Dict[str, Any]:
        """
        Explains an individual prediction by computing positive (risk-increasing)
        and negative (risk-decreasing) SHAP attributions.
        """
        X_scaled = self.preprocessor.transform(X_single)
        
        # Calculate prediction probability
        if hasattr(self.estimator, "predict_proba"):
            proba = float(self.estimator.predict_proba(X_scaled)[0, 1])
        else:
            proba = float(self.estimator.predict(X_scaled)[0])

        if self.explainer is not None:
            shap_values = self.explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):
                # Binary classification list [class 0, class 1]
                vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 3:
                    vals = shap_values[0, :, 1]
                elif shap_values.ndim == 2:
                    vals = shap_values[0]
                else:
                    vals = shap_values
            else:
                vals = np.zeros(len(self.feature_names))
        else:
            # Fallback based on normalized feature importance
            if hasattr(self.estimator, "feature_importances_"):
                importances = self.estimator.feature_importances_
                vals = importances * (X_scaled[0] - np.mean(X_scaled[0]))
            else:
                vals = np.zeros(len(self.feature_names))

        # Format attributions
        attributions = []
        for i, val in enumerate(vals):
            feat = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            raw_val = X_single[feat].values[0] if feat in X_single.columns else X_scaled[0, i]
            attributions.append({
                "feature": feat,
                "shap_value": float(val),
                "feature_value": "N/A" if (raw_val is None or pd.isna(raw_val)) else (round(float(raw_val), 3) if isinstance(raw_val, (int, float, np.number)) else str(raw_val)),
                "impact": "INCREASES_RISK" if val > 0 else "DECREASES_RISK"
            })

        # Sort top positive and negative drivers
        positive_drivers = sorted([a for a in attributions if a["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)[:top_k]
        negative_drivers = sorted([a for a in attributions if a["shap_value"] < 0], key=lambda x: x["shap_value"])[:top_k]

        return {
            "prediction_probability": round(proba, 4),
            "top_risk_increasing_factors": positive_drivers,
            "top_risk_decreasing_factors": negative_drivers,
            "all_attributions": attributions,
            "scientific_note": "SHAP values quantify statistical feature contribution to model score. They represent model behavior and do not constitute physical or causal proof."
        }

    def generate_global_summary_plot(
        self,
        X_sample: pd.DataFrame,
        output_filename: str = "shap_global_summary.png",
        max_display: int = 15
    ) -> Optional[str]:
        """
        Generates and saves a global SHAP feature importance summary chart.
        """
        if self.explainer is None or not SHAP_AVAILABLE:
            return None

        X_scaled = self.preprocessor.transform(X_sample)
        shap_values = self.explainer.shap_values(X_scaled)
        
        if isinstance(shap_values, list) and len(shap_values) > 1:
            vals = shap_values[1]
        else:
            vals = shap_values

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            vals,
            X_scaled,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title("Global Feature Importance (TreeSHAP Attributions)", fontsize=14, pad=15)
        plt.tight_layout()
        
        fig_path = os.path.join(FIGURES_DIR, output_filename)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        return fig_path
