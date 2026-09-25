"""
Unit tests for domain-physics feature engineering and sensor spike clipping.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engine import (
    clip_sensor_spikes,
    engineer_physics_features,
    get_feature_matrix,
    PHYSICS_FEATURES
)


def test_clip_sensor_spikes():
    df_raw = pd.DataFrame({
        "cell_temperature_max": [150.0, -50.0, 35.0],
        "state_of_health": [120.0, -10.0, 85.0],
        "fast_charge_ratio": [1.5, -0.2, 0.4],
        "internal_resistance": [10.0, -1.0, 0.35]
    })
    
    df_clean = clip_sensor_spikes(df_raw)
    
    assert df_clean["cell_temperature_max"].max() <= 85.0
    assert df_clean["cell_temperature_max"].min() >= -30.0
    assert df_clean["state_of_health"].max() <= 100.0
    assert df_clean["state_of_health"].min() >= 0.0
    assert df_clean["fast_charge_ratio"].max() <= 1.0
    assert df_clean["fast_charge_ratio"].min() >= 0.0
    assert df_clean["internal_resistance"].min() >= 0.01


def test_engineer_physics_features():
    df_sample = pd.DataFrame({
        "average_ambient_temperature": [25.0, 45.0],
        "cell_temperature_max": [30.0, 55.0],
        "cell_temperature_min": [25.0, 40.0],
        "cell_voltage_max": [4.1, 4.2],
        "cell_voltage_min": [4.08, 4.10],
        "fast_charge_ratio": [0.2, 0.8],
        "cycle_count": [500, 2000],
        "internal_resistance": [0.25, 0.60],
        "capacity_loss_percent": [5.0, 25.0],
        "odometer_km": [50000, 150000],
        "thermal_health_score": [95.0, 50.0],
        "charging_quality_score": [90.0, 60.0],
        "driving_stress_score": [30.0, 70.0],
        "battery_health_percent": [95.0, 75.0],
        "cooling_system_health": [90.0, 40.0],
        "battery_capacity_kwh": [75.0, 60.0],
        "state_of_charge": [60.0, 80.0],
        "vehicle_age_years": [2.0, 6.0],
    })
    
    df_feat = engineer_physics_features(df_sample)
    
    # Verify core features exist and are not NaN
    for feat in [
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
        'degradation_acceleration_index'
    ]:
        assert feat in df_feat.columns
        assert not df_feat[feat].isna().any()

    # Arrhenius factor for 55°C must be strictly higher than 30°C
    assert df_feat.loc[1, 'arrhenius_aging_factor'] > df_feat.loc[0, 'arrhenius_aging_factor']


def test_get_feature_matrix_leakage_safety():
    df = pd.DataFrame({
        "vehicle_id": ["EV001", "EV002"],
        "battery_serial": ["BAT01", "BAT02"],
        "battery_failure": [0, 1],
        "state_of_health": [92.0, 68.0],
        "predicted_remaining_life_cycles": [8000.0, 2500.0],
        "cell_voltage_avg": [3.7, 3.5],
        "cell_temperature_max": [32.0, 58.0],
        "cycle_count": [400, 2200]
    })
    
    # 1. Failure Prediction Matrix
    X_fail, y_fail, cols_fail = get_feature_matrix(df, target_type="failure")
    assert "vehicle_id" not in X_fail.columns
    assert "battery_serial" not in X_fail.columns
    assert "battery_failure" not in X_fail.columns
    assert y_fail is not None
    
    # 2. SOH Prediction Matrix (must exclude SOH target & duplicates)
    X_soh, y_soh, cols_soh = get_feature_matrix(df, target_type="soh")
    assert "state_of_health" not in X_soh.columns
    assert "battery_health_percent" not in X_soh.columns
    assert "battery_failure" not in X_soh.columns
    assert y_soh is not None

    # 3. RUL Prediction Matrix (must exclude RUL target)
    X_rul, y_rul, cols_rul = get_feature_matrix(df, target_type="rul")
    assert "predicted_remaining_life_cycles" not in X_rul.columns
    assert "battery_failure" not in X_rul.columns
    assert y_rul is not None
