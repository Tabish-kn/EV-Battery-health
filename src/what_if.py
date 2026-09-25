"""
What-If Counterfactual & Sensitivity Simulation Engine.
Enables engineers and fleet operators to simulate telemetry alterations (e.g. ambient temperature shifts,
fast-charging frequency, cooling degradation, cycle aging) and observe multidimensional impact
across Failure Probability, State of Health (SOH), Remaining Useful Life (RUL), and Composite Risk.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.models import ModelBundle
from src.feature_engine import get_feature_matrix, clip_sensor_spikes, engineer_physics_features
from src.risk_scoring import compute_battery_risk_score


class BatteryWhatIfSimulator:
    """
    Simulation engine that accepts a baseline telemetry record, applies parameter shifts,
    recalculates physics features, and runs all model pipelines to evaluate scenario deltas.
    """

    def __init__(
        self,
        failure_bundle: ModelBundle,
        soh_bundle: Optional[ModelBundle] = None,
        rul_bundle: Optional[ModelBundle] = None,
        anomaly_bundle: Optional[ModelBundle] = None
    ):
        self.failure_bundle = failure_bundle
        self.soh_bundle = soh_bundle
        self.rul_bundle = rul_bundle
        self.anomaly_bundle = anomaly_bundle

    def _predict_pipeline_state(self, df_row: pd.DataFrame) -> Dict[str, Any]:
        """Runs predictions across all loaded model bundles for a single row."""
        # 1. Feature Engineering
        df_clean = clip_sensor_spikes(df_row)
        df_feat = engineer_physics_features(df_clean)

        # 2. Failure Probability
        X_fail, _, _ = get_feature_matrix(df_row, target_type="failure")
        fail_proba = float(self.failure_bundle.predict_proba(X_fail)[0, 1])

        # 3. SOH Prediction
        if self.soh_bundle:
            X_soh, _, _ = get_feature_matrix(df_row, target_type="soh")
            soh_pred = float(np.clip(self.soh_bundle.predict(X_soh)[0], 0.0, 100.0))
        else:
            soh_pred = float(df_row.get("state_of_health", 85.0))

        # 4. RUL Prediction
        if self.rul_bundle:
            X_rul, _, _ = get_feature_matrix(df_row, target_type="rul")
            rul_pred = float(np.maximum(0.0, self.rul_bundle.predict(X_rul)[0]))
        else:
            rul_pred = float(df_row.get("predicted_remaining_life_cycles", 5000.0))

        # 5. Anomaly Score
        if self.anomaly_bundle:
            X_anom, _, _ = get_feature_matrix(df_row, target_type="anomaly")
            X_anom_scaled = self.anomaly_bundle.preprocessor.transform(X_anom)
            anom_score = float(self.anomaly_bundle.estimator.calculate_anomaly_score_0_100(X_anom_scaled)[0])
        else:
            anom_score = 15.0

        # 6. Risk Score
        cell_temp_max = float(df_feat.get("cell_temperature_max", [35.0])[0])
        voltage_imb = float(df_feat.get("voltage_imbalance", [15.0])[0])
        fast_chg = float(df_feat.get("fast_charge_ratio", [0.3])[0])
        cooling = float(df_feat.get("cooling_system_health", [80.0])[0])
        cycles = float(df_feat.get("cycle_count", [1000.0])[0])

        risk_data = compute_battery_risk_score(
            failure_probability=fail_proba,
            soh_percent=soh_pred,
            cell_temp_max=cell_temp_max,
            voltage_imbalance_mv=voltage_imb,
            fast_charge_ratio=fast_chg,
            anomaly_score_100=anom_score,
            cooling_health=cooling,
            cycle_count=cycles
        )

        return {
            "failure_probability": fail_proba,
            "state_of_health_pct": soh_pred,
            "remaining_useful_life_cycles": rul_pred,
            "anomaly_score_100": anom_score,
            "composite_risk_score": risk_data["composite_risk_score"],
            "risk_category": risk_data["risk_category"],
            "sub_scores": risk_data["sub_scores"]
        }

    def simulate_scenario(
        self,
        baseline_record: pd.DataFrame,
        parameter_modifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes scenario simulation by updating specified columns and calculating deltas.
        
        Args:
            baseline_record: 1-row DataFrame containing baseline telemetry.
            parameter_modifications: Dict of {column_name: new_value}.
            
        Returns:
            Dict containing baseline state, simulated state, and deltas.
        """
        baseline_state = self._predict_pipeline_state(baseline_record.copy())

        # Construct simulated record
        simulated_record = baseline_record.copy()
        for col, val in parameter_modifications.items():
            simulated_record[col] = val

        simulated_state = self._predict_pipeline_state(simulated_record)

        # Calculate deltas
        deltas = {
            "failure_probability_delta_pct_points": round((simulated_state["failure_probability"] - baseline_state["failure_probability"]) * 100.0, 2),
            "soh_delta_pct": round(simulated_state["state_of_health_pct"] - baseline_state["state_of_health_pct"], 2),
            "rul_delta_cycles": round(simulated_state["remaining_useful_life_cycles"] - baseline_state["remaining_useful_life_cycles"], 0),
            "composite_risk_delta_points": round(simulated_state["composite_risk_score"] - baseline_state["composite_risk_score"], 1),
        }

        return {
            "parameters_modified": parameter_modifications,
            "baseline": baseline_state,
            "simulated": simulated_state,
            "deltas": deltas,
            "scientific_disclaimer": "Simulated outputs represent model interpolations based on statistical associations. They are for exploratory analysis and do not guarantee physical cell outcomes."
        }
