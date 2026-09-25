"""
Multi-Factor Battery Risk Scoring & Diagnostic Index Engine.
Computes a normalized 0-100 composite risk index composed of six diagnostic sub-indices:
1. Failure Probability Risk
2. Thermal Stress Risk
3. Electrical & Voltage Imbalance Risk
4. Charging & Fast-Charge Stress Risk
5. Degradation & Aging Risk
6. Multivariate Telemetry Anomaly Risk
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from src.config import RISK_WEIGHTS


def compute_battery_risk_score(
    failure_probability: float,
    soh_percent: float,
    cell_temp_max: float,
    voltage_imbalance_mv: float,
    fast_charge_ratio: float,
    anomaly_score_100: float,
    cooling_health: float = 80.0,
    cycle_count: float = 1000.0,
    weights: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Calculates a comprehensive, normalized 0-100 diagnostic risk score.
    
    Sub-Index Formulations:
    - Thermal Risk: f(cell_temp_max, cooling_health)
    - Electrical Risk: f(voltage_imbalance_mv)
    - Charging Stress: f(fast_charge_ratio)
    - Degradation Risk: f(100 - soh_percent, cycle_count)
    - Anomaly Risk: anomaly_score_100
    - Failure Risk: failure_probability * 100
    """
    weights = weights or RISK_WEIGHTS

    # 1. Failure Probability Risk (0 - 100)
    sub_failure = np.clip(failure_probability * 100.0, 0.0, 100.0)

    # 2. Thermal Stress Risk (0 - 100)
    # Thresholds: > 45°C begins elevated risk, > 60°C is critical, degraded cooling amplifies
    temp_excess = max(0.0, cell_temp_max - 35.0)
    cooling_penalty = max(0.0, (80.0 - cooling_health) * 0.5)
    sub_thermal = np.clip((temp_excess * 2.2) + cooling_penalty, 0.0, 100.0)

    # 3. Electrical Risk (0 - 100)
    # Voltage imbalance: < 15mV is nominal, > 50mV is elevated, > 80mV is critical
    sub_electrical = np.clip((voltage_imbalance_mv / 80.0) * 100.0, 0.0, 100.0)

    # 4. Charging Stress (0 - 100)
    # Fast charge ratio > 0.4 accelerates lithium plating and mechanical SEI cracking
    sub_charging = np.clip(fast_charge_ratio * 100.0, 0.0, 100.0)

    # 5. Degradation / Aging Risk (0 - 100)
    # SOH fade: 100% -> 0 risk; 70% SOH -> 100 risk
    soh_fade = max(0.0, 100.0 - soh_percent)
    sub_degradation = np.clip((soh_fade / 30.0) * 100.0, 0.0, 100.0)

    # 6. Anomaly Risk (0 - 100)
    sub_anomaly = np.clip(anomaly_score_100, 0.0, 100.0)

    # Weighted Composite Score
    composite_score = (
        weights["failure_prob"] * sub_failure +
        weights["thermal_risk"] * sub_thermal +
        weights["electrical_risk"] * sub_electrical +
        weights["charging_stress"] * sub_charging +
        weights["degradation_risk"] * sub_degradation +
        weights["anomaly_risk"] * sub_anomaly
    )
    composite_score = float(np.clip(composite_score, 0.0, 100.0))

    # Determine Severity Category
    if composite_score < 30.0:
        category = "LOW"
        color = "#10b981"  # Emerald Green
    elif composite_score < 60.0:
        category = "MODERATE"
        color = "#f59e0b"  # Amber Orange
    elif composite_score < 80.0:
        category = "HIGH"
        color = "#ef4444"  # Red
    else:
        category = "CRITICAL"
        color = "#b91c1c"  # Dark Red

    def _sub_label(score: float) -> str:
        if score < 30.0:
            return "LOW"
        elif score < 60.0:
            return "MEDIUM"
        else:
            return "HIGH"

    return {
        "composite_risk_score": round(composite_score, 1),
        "risk_category": category,
        "color": color,
        "sub_scores": {
            "failure_risk": round(float(sub_failure), 1),
            "thermal_risk": round(float(sub_thermal), 1),
            "electrical_risk": round(float(sub_electrical), 1),
            "charging_risk": round(float(sub_charging), 1),
            "degradation_risk": round(float(sub_degradation), 1),
            "anomaly_risk": round(float(sub_anomaly), 1)
        },
        "sub_labels": {
            "failure_risk": _sub_label(sub_failure),
            "thermal_risk": _sub_label(sub_thermal),
            "electrical_risk": _sub_label(sub_electrical),
            "charging_risk": _sub_label(sub_charging),
            "degradation_risk": _sub_label(sub_degradation),
            "anomaly_risk": _sub_label(sub_anomaly)
        },
        "weights": weights,
        "disclaimer": "Model-derived composite diagnostic indicator. Does not constitute a deterministic warranty or physical safety certification."
    }
