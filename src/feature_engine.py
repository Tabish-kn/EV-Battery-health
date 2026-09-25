"""
Feature Engineering Engine for EV Battery Health, Degradation & Diagnostics.
Derives domain-physics features, electrochemical degradation indicators, thermal strain,
electrical stress, and BMS fault density metrics while strictly preventing target leakage.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import COLUMN_ALIASES, ID_COLUMNS, TARGET_FAILURE, TARGET_SOH, TARGET_SOH_ALT, TARGET_RUL


# Canonical Physics & Electrochemical Feature Names
PHYSICS_FEATURES = [
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
    'c_rate_proxy',
    'dod_fast_charge_stress',
    'electrical_efficiency_loss',
    'thermal_severity_index',
    'bms_fault_density'
]


def get_series(df: pd.DataFrame, alias_key: str, default_val=np.nan) -> pd.Series:
    """Retrieve a series by resolving candidate alias column names."""
    candidates = COLUMN_ALIASES.get(alias_key, [alias_key])
    for col in candidates:
        if col in df.columns:
            return df[col].copy()
    return pd.Series(default_val, index=df.index, dtype=float)


def clip_sensor_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clips sensor spikes to physically realistic physical boundaries:
    - Temperatures: [-30°C, 85°C]
    - State of Health (SoH) and Health Percent: [0%, 100%]
    - Efficiencies & Ratios: [0.0, 1.0] or [0%, 100%]
    - Cell voltages: [2.0V, 4.5V]
    - Internal resistance: [0.01 mOhm, 5.0 mOhm]
    """
    df_clean = df.copy()

    # Temperature columns
    temp_cols = [
        'cell_temperature_avg', 'cell_temperature_max', 'cell_temperature_min',
        'average_ambient_temperature', 'maximum_temperature', 'minimum_temperature'
    ]
    for col in temp_cols:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].clip(lower=-30.0, upper=85.0)

    # Health & SOH columns
    soh_cols = ['battery_health_percent', 'battery_health_percentage', 'state_of_health']
    for col in soh_cols:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].clip(lower=0.0, upper=100.0)

    # State of charge and Depth of discharge
    for col in ['state_of_charge', 'depth_of_discharge', 'capacity_loss_percent', 'capacity_loss_percentage']:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].clip(lower=0.0, upper=100.0)

    # Charging ratios
    ratio_cols = [
        'fast_charge_ratio', 'slow_charge_ratio', 'overnight_charging_ratio', 'home_charging_ratio',
        'highway_driving_ratio', 'city_driving_ratio'
    ]
    for col in ratio_cols:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].clip(lower=0.0, upper=1.0)

    # Internal resistance non-negative
    for col in ['internal_resistance', 'internal_resistance_mOhm']:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].clip(lower=0.01, upper=5.0)

    return df_clean


def engineer_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives 12 Core Domain-Physics & Novel Electrochemical features + 5 Extended BMS/Thermal indicators.

    1. Arrhenius Aging Factor:
       np.exp(np.maximum(0, np.maximum(average_ambient_temp, cell_temperature_max) - 25) / 10.0)
    2. Cell Temperature Variance Delta:
       cell_temperature_max - cell_temperature_min
    3. Cell Voltage Spread:
       cell_voltage_max - cell_voltage_min (or inferred from std/imbalance)
    4. Fast-Charge Cumulative Stress Index:
       fast_charge_ratio * cycle_count
    5. Resistance Growth Proxy:
       internal_resistance * (1 + capacity_loss_percent / 100.0)
    6. Equivalent Full Cycles (EFC) per 1,000 km:
       (cycle_count / np.maximum(odometer_km, 1.0)) * 1000
    7. Composite Physical Stress Aggregate:
       thermal_stress + charging_stress + driving_stress
    8. SoH Wear Loss per Cycle:
       (100 - battery_health_percentage) / np.maximum(cycle_count, 1)
    9. Thermal Dissipation Strain:
       (cell_temperature_max - ambient_temp) / np.maximum(cooling_system_health, 1.0)
    10. Internal Resistance per kWh Capacity:
        internal_resistance / np.maximum(battery_capacity_kwh, 1.0)
    11. Lithium Plating Susceptibility Index:
        np.maximum(0.0, 15.0 - ambient_temp) * fast_charge_ratio * (state_of_charge / 100.0)
    12. Degradation Acceleration Ratio:
        capacity_loss_percent / np.maximum(vehicle_age_years, 0.1)

    Extended Domain Indicators:
    13. C-Rate Proxy:
        average_charge_power_kw / np.maximum(battery_capacity_kwh, 1.0)
    14. DOD-Fast Charge Stress:
        depth_of_discharge * (1.0 + fast_charge_ratio)
    15. Electrical Efficiency Loss:
        (100.0 - charge_efficiency) + (100.0 - discharge_efficiency)
    16. Thermal Severity Index:
        cell_temperature_max * (1.0 + thermal_runaway_risk / 100.0)
    17. BMS Fault Density:
        (BMS_warning_count + sensor_fault_count + abnormal_voltage_events) / np.maximum(vehicle_age_years, 0.5)
    """
    df_feat = df.copy()

    # Extract clean series with fallback resolution
    ambient_temp = get_series(df_feat, 'ambient_temp', default_val=25.0).fillna(25.0)
    cell_temp_max = get_series(df_feat, 'cell_temp_max', default_val=35.0).fillna(35.0)
    cell_temp_avg = get_series(df_feat, 'cell_temp_avg', default_val=30.0).fillna(30.0)

    # 1. Arrhenius Aging Factor
    effective_temp = np.maximum(ambient_temp, cell_temp_max)
    df_feat['arrhenius_aging_factor'] = np.exp(np.maximum(0.0, effective_temp - 25.0) / 10.0)

    # 2. Cell Temperature Variance Delta
    if 'cell_temperature_min' in df_feat.columns and df_feat['cell_temperature_min'].notna().any():
        cell_temp_min = df_feat['cell_temperature_min'].fillna(25.0)
    elif 'temperature_variance' in df_feat.columns and df_feat['temperature_variance'].notna().any():
        cell_temp_min = cell_temp_max - df_feat['temperature_variance'].fillna(5.0)
    else:
        cell_temp_min = cell_temp_avg - np.maximum(0.0, cell_temp_max - cell_temp_avg)
    df_feat['cell_temperature_variance_delta'] = (cell_temp_max - cell_temp_min).clip(lower=0.0)

    # 3. Cell Voltage Spread
    if 'cell_voltage_max' in df_feat.columns and 'cell_voltage_min' in df_feat.columns:
        df_feat['cell_voltage_spread'] = (df_feat['cell_voltage_max'] - df_feat['cell_voltage_min']).clip(lower=0.0)
    elif 'voltage_imbalance' in df_feat.columns:
        df_feat['cell_voltage_spread'] = (df_feat['voltage_imbalance'].fillna(20.0) / 1000.0).clip(lower=0.0)
    elif 'cell_voltage_std' in df_feat.columns:
        df_feat['cell_voltage_spread'] = (df_feat['cell_voltage_std'].fillna(0.01) * 6.0).clip(lower=0.0)
    else:
        df_feat['cell_voltage_spread'] = 0.020

    # 4. Fast-Charge Cumulative Stress Index
    fast_charge_ratio = get_series(df_feat, 'fast_charge_ratio', default_val=0.3).fillna(0.3)
    cycle_count = get_series(df_feat, 'cycle_count', default_val=1000.0).fillna(1000.0)
    df_feat['fast_charge_cumulative_stress'] = fast_charge_ratio * cycle_count

    # 5. Resistance Growth Proxy
    ir = get_series(df_feat, 'internal_resistance', default_val=0.4).fillna(0.4)
    cap_loss = get_series(df_feat, 'capacity_loss', default_val=15.0).fillna(15.0)
    df_feat['resistance_growth_proxy'] = ir * (1.0 + cap_loss / 100.0)

    # 6. Equivalent Full Cycles (EFC) per 1,000 km
    odometer_km = get_series(df_feat, 'odometer_km', default_val=100000.0).fillna(100000.0)
    df_feat['efc_per_1000km'] = (cycle_count / np.maximum(odometer_km, 1.0)) * 1000.0

    # 7. Composite Physical Stress Aggregate
    if 'thermal_stress_score' in df_feat.columns and df_feat['thermal_stress_score'].notna().any():
        thermal_stress = df_feat['thermal_stress_score'].fillna(30.0)
    else:
        th_score = get_series(df_feat, 'thermal_health_score', default_val=80.0).fillna(80.0)
        thermal_stress = 100.0 - th_score

    if 'charging_stress_score' in df_feat.columns and df_feat['charging_stress_score'].notna().any():
        charging_stress = df_feat['charging_stress_score'].fillna(30.0)
    else:
        cq_score = get_series(df_feat, 'charging_quality_score', default_val=80.0).fillna(80.0)
        charging_stress = 100.0 - cq_score

    driving_stress = get_series(df_feat, 'driving_stress_score', default_val=40.0).fillna(40.0)
    df_feat['composite_physical_stress'] = thermal_stress + charging_stress + driving_stress

    # 8. SoH Wear Loss per Cycle
    soh_pct = get_series(df_feat, 'battery_health_percent', default_val=85.0).fillna(85.0)
    df_feat['soh_wear_loss_per_cycle'] = (100.0 - soh_pct) / np.maximum(cycle_count, 1.0)

    # 9. Thermal Dissipation Strain
    cooling_health = get_series(df_feat, 'cooling_system_health', default_val=80.0).fillna(80.0)
    df_feat['thermal_dissipation_strain'] = (
        (cell_temp_max - ambient_temp) / np.maximum(cooling_health, 1.0)
    )

    # 10. Internal Resistance per kWh Capacity
    capacity_kwh = get_series(df_feat, 'battery_capacity_kwh', default_val=75.0).fillna(75.0)
    df_feat['ir_per_kwh'] = ir / np.maximum(capacity_kwh, 1.0)

    # 11. Lithium Plating Susceptibility Metric
    soc = get_series(df_feat, 'state_of_charge', default_val=50.0).fillna(50.0)
    df_feat['lithium_plating_risk'] = (
        np.maximum(0.0, 15.0 - ambient_temp) * fast_charge_ratio * (soc / 100.0)
    )

    # 12. Degradation Acceleration Index
    vehicle_age = get_series(df_feat, 'vehicle_age_years', default_val=5.0).fillna(5.0)
    df_feat['degradation_acceleration_index'] = cap_loss / np.maximum(vehicle_age, 0.1)

    # Extended Physics Metrics
    # 13. C-Rate Proxy
    charge_power = get_series(df_feat, 'average_charge_power_kw', default_val=50.0).fillna(50.0)
    df_feat['c_rate_proxy'] = charge_power / np.maximum(capacity_kwh, 1.0)

    # 14. DOD-Fast Charge Stress
    dod = df_feat['depth_of_discharge'].fillna(50.0) if 'depth_of_discharge' in df_feat.columns else pd.Series(50.0, index=df_feat.index)
    df_feat['dod_fast_charge_stress'] = dod * (1.0 + fast_charge_ratio)

    # 15. Electrical Efficiency Loss
    chg_eff = df_feat['charge_efficiency'].fillna(90.0) if 'charge_efficiency' in df_feat.columns else pd.Series(90.0, index=df_feat.index)
    dis_eff = df_feat['discharge_efficiency'].fillna(90.0) if 'discharge_efficiency' in df_feat.columns else pd.Series(90.0, index=df_feat.index)
    df_feat['electrical_efficiency_loss'] = (100.0 - chg_eff) + (100.0 - dis_eff)

    # 16. Thermal Severity Index
    runaway_risk = df_feat['thermal_runaway_risk'].fillna(20.0) if 'thermal_runaway_risk' in df_feat.columns else pd.Series(20.0, index=df_feat.index)
    df_feat['thermal_severity_index'] = cell_temp_max * (1.0 + runaway_risk / 100.0)

    # 17. BMS Fault Density
    bms_warn = df_feat['BMS_warning_count'].fillna(0.0) if 'BMS_warning_count' in df_feat.columns else pd.Series(0.0, index=df_feat.index)
    sens_flt = df_feat['sensor_fault_count'].fillna(0.0) if 'sensor_fault_count' in df_feat.columns else pd.Series(0.0, index=df_feat.index)
    abn_volt = df_feat['abnormal_voltage_events'].fillna(0.0) if 'abnormal_voltage_events' in df_feat.columns else pd.Series(0.0, index=df_feat.index)
    df_feat['bms_fault_density'] = (bms_warn + sens_flt + abn_volt) / np.maximum(vehicle_age, 0.5)

    return df_feat


def get_feature_matrix(
    df: pd.DataFrame,
    target_type: str = "failure"
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    """
    Extracts feature matrix X and target y for the specified task while strictly
    dropping identifiers and preventing target leakage.

    Args:
        df: Raw or partially cleaned DataFrame.
        target_type: One of 'failure', 'soh', 'rul', 'anomaly'.

    Returns:
        X (DataFrame), y (Series or None), feature_names (List[str])
    """
    # 1. Clip spikes & derive physics features
    df_clean = clip_sensor_spikes(df)
    df_feat = engineer_physics_features(df_clean)

    # 2. Determine target and leakage columns to exclude
    exclude_cols = list(ID_COLUMNS)

    if target_type == "failure":
        target_col = TARGET_FAILURE
        # Do not include the target in X
        exclude_cols.append(TARGET_FAILURE)
        # SOH and RUL are available telemetry in this dataset if predicting failure
    elif target_type == "soh":
        target_col = TARGET_SOH if TARGET_SOH in df_feat.columns else TARGET_SOH_ALT
        # Exclude direct SOH duplicates / leakages
        exclude_cols.extend([TARGET_SOH, TARGET_SOH_ALT, 'battery_health_percentage', 'capacity_loss_percent', 'capacity_loss_percentage', 'soh_wear_loss_per_cycle', 'resistance_growth_proxy', 'degradation_acceleration_index', TARGET_FAILURE])
    elif target_type == "rul":
        target_col = TARGET_RUL
        # Exclude RUL and failure
        exclude_cols.extend([TARGET_RUL, TARGET_FAILURE])
    elif target_type == "anomaly":
        target_col = None
        exclude_cols.extend([TARGET_FAILURE, TARGET_SOH, TARGET_SOH_ALT, TARGET_RUL])
    else:
        raise ValueError(f"Unknown target_type: '{target_type}'. Expected 'failure', 'soh', 'rul', or 'anomaly'.")

    # Extract target series if present
    y = None
    if target_col and target_col in df_feat.columns:
        y = df_feat[target_col].copy()

    # Feature matrix X
    drop_candidates = [c for c in exclude_cols if c in df_feat.columns]
    X = df_feat.drop(columns=drop_candidates)

    feature_names = X.columns.tolist()
    return X, y, feature_names
