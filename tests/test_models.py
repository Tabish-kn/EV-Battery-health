"""
Unit tests for model bundling, risk scoring, and anomaly detection logic.
"""

import numpy as np
import pandas as pd
import pytest

from src.risk_scoring import compute_battery_risk_score
from src.anomaly_detection import BatteryAnomalyDetector


def test_compute_battery_risk_score_ranges():
    # Test nominal healthy battery
    res_healthy = compute_battery_risk_score(
        failure_probability=0.02,
        soh_percent=95.0,
        cell_temp_max=32.0,
        voltage_imbalance_mv=8.0,
        fast_charge_ratio=0.15,
        anomaly_score_100=10.0,
        cooling_health=95.0
    )
    
    assert 0.0 <= res_healthy["composite_risk_score"] <= 100.0
    assert res_healthy["risk_category"] == "LOW"
    assert res_healthy["color"] == "#10b981"
    
    # Test critical degraded battery
    res_critical = compute_battery_risk_score(
        failure_probability=0.88,
        soh_percent=55.0,
        cell_temp_max=68.0,
        voltage_imbalance_mv=95.0,
        fast_charge_ratio=0.85,
        anomaly_score_100=92.0,
        cooling_health=25.0
    )
    
    assert res_critical["composite_risk_score"] >= 70.0
    assert res_critical["risk_category"] in ["HIGH", "CRITICAL"]


def test_battery_anomaly_detector():
    np.random.seed(42)
    # Generate 200 normal samples and 10 extreme outliers
    X_normal = np.random.normal(loc=0.0, scale=1.0, size=(200, 5))
    X_outliers = np.random.uniform(low=8.0, high=15.0, size=(10, 5))
    X = np.vstack([X_normal, X_outliers])
    
    detector = BatteryAnomalyDetector(contamination=0.05, random_state=42)
    detector.fit(X)
    
    scores = detector.calculate_anomaly_score_0_100(X)
    assert len(scores) == 210
    assert (scores >= 0.0).all() and (scores <= 100.0).all()
    
    # Outliers should have higher anomaly scores than normal samples
    assert np.mean(scores[200:]) > np.mean(scores[:200])
