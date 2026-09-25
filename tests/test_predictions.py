"""
Unit tests for data validation, batch prediction, and what-if simulation workflows.
"""

import numpy as np
import pandas as pd
import pytest

from src.validation import validate_dataset, clean_dataset, check_target_leakage, generate_data_quality_report
from src.degradation import calculate_arrhenius_acceleration, simulate_degradation_trajectory


def test_data_validation_and_cleaning():
    df = pd.DataFrame({
        "cell_voltage_avg": [3.7, 4.8, 2.1], # 4.8 exceeds upper bound 4.5
        "state_of_health": [90.0, 110.0, 80.0], # 110 exceeds 100
        "vehicle_id": ["V1", "V2", "V3"]
    })
    
    val_report = validate_dataset(df)
    assert val_report["valid"] is True
    assert "cell_voltage_avg" in val_report["boundary_violations"]
    assert "state_of_health" in val_report["boundary_violations"]
    
    # Clean dataset
    df_clean = clean_dataset(df)
    assert df_clean["cell_voltage_avg"].max() <= 4.5
    assert df_clean["state_of_health"].max() <= 100.0


def test_target_leakage_detector():
    df = pd.DataFrame({
        "battery_failure": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "leaked_failure_copy": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], # 100% correlation
        "cell_voltage": [3.8, 3.6, 3.9, 3.5, 3.8, 3.4, 3.7, 3.5, 3.9, 3.6]
    })
    
    leaks = check_target_leakage(df, target_col="battery_failure", correlation_threshold=0.95)
    assert len(leaks) >= 1
    assert leaks[0]["feature"] == "leaked_failure_copy"
    assert leaks[0]["risk"] == "HIGH_TARGET_LEAKAGE"


def test_degradation_kinetics():
    # Arrhenius acceleration for hot vs cold
    acc_hot = calculate_arrhenius_acceleration(45.0, reference_temp_celsius=25.0)
    acc_nominal = calculate_arrhenius_acceleration(25.0, reference_temp_celsius=25.0)
    
    assert acc_hot > acc_nominal
    assert abs(acc_nominal - 1.0) < 0.05
    
    # Trajectory simulation
    traj = simulate_degradation_trajectory(
        current_soh=90.0,
        current_cycle=500,
        fast_charge_ratio=0.4,
        avg_temperature_c=35.0,
        project_cycles=2000
    )
    
    assert len(traj["projected_soh"]) == 100
    assert traj["projected_soh"][0] == 90.0
    assert traj["projected_soh"][-1] < 90.0
    assert traj["estimated_cycles_to_eol"] > 0


def test_batch_pipeline_and_models():
    from src.models import ModelBundle
    from src.utils import run_batch_pipeline
    from src.config import FAILURE_MODEL_DIR, SOH_MODEL_DIR, RUL_MODEL_DIR, ANOMALY_MODEL_DIR
    from src.data_loader import get_sample_data
    
    sample_df = get_sample_data(10)
    failure_bundle = ModelBundle.load(str(FAILURE_MODEL_DIR))
    soh_bundle = ModelBundle.load(str(SOH_MODEL_DIR))
    rul_bundle = ModelBundle.load(str(RUL_MODEL_DIR))
    anomaly_bundle = ModelBundle.load(str(ANOMALY_MODEL_DIR))
    
    enriched = run_batch_pipeline(sample_df, failure_bundle, soh_bundle, rul_bundle, anomaly_bundle)
    
    assert "predicted_failure_probability" in enriched.columns
    assert "predicted_soh_percent" in enriched.columns
    assert "predicted_rul_cycles" in enriched.columns
    assert "anomaly_score_100" in enriched.columns
    assert "battery_risk_score" in enriched.columns
    assert "risk_category" in enriched.columns
    assert len(enriched) == 10
