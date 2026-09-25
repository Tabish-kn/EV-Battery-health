"""
Feature Engineering Engine for EV Battery Health & Powertrain Diagnostics.
Handles raw telemetry columns, sensor spike clipping, domain-physics features,
and novel electrochemical degradation indicators.
"""

import numpy as np
import pandas as pd


# Column alias dictionary to support both raw CSV and prompt naming variations
COLUMN_ALIASES = {
    'ambient_temp': ['average_ambient_temperature', 'average_ambient_temp', 'ambient_temperature'],
    'ambient_temp_max': ['maximum_temperature', 'maximum_ambient_temp'],
    'ambient_temp_min': ['minimum_temperature', 'minimum_ambient_temp'],
    'cell_temp_max': ['cell_temperature_max', 'cell_temp_max'],
    'cell_temp_avg': ['cell_temperature_avg', 'cell_temp_avg'],
    'cell_temp_min': ['cell_temperature_min', 'cell_temp_min'],
    'temp_variance': ['temperature_variance', 'cell_temp_variance'],
    'cell_voltage_avg': ['cell_voltage_avg', 'cell_voltage_mean'],
    'cell_voltage_std': ['cell_voltage_std', 'cell_voltage_sigma'],
    'cell_voltage_max': ['cell_voltage_max'],
    'cell_voltage_min': ['cell_voltage_min'],
    'voltage_imbalance': ['voltage_imbalance', 'cell_voltage_imbalance'],
    'pack_voltage': ['pack_voltage'],
    'cycle_count': ['cycle_count', 'charging_cycles'],
    'fast_charge_ratio': ['fast_charge_ratio'],
    'slow_charge_ratio': ['slow_charge_ratio'],
    'internal_resistance': ['internal_resistance', 'internal_resistance_mOhm', 'battery_internal_resistance'],
    'capacity_loss': ['capacity_loss_percent', 'capacity_loss_percentage'],
    'soh': ['state_of_health', 'battery_health_percent', 'battery_health_percentage'],
    'battery_health_percent': ['battery_health_percent', 'battery_health_percentage', 'state_of_health'],
    'odometer_km': ['odometer_km', 'odometer'],
    'thermal_health_score': ['thermal_health_score', 'thermal_stress_score'],
    'charging_quality_score': ['charging_quality_score', 'charging_stress_score'],
    'driving_stress_score': ['driving_stress_score'],
    'cooling_system_health': ['cooling_system_health'],
    'battery_capacity_kwh': ['battery_capacity_kwh', 'nominal_capacity_kwh'],
    'state_of_charge': ['state_of_charge', 'soc'],
    'vehicle_age_years': ['vehicle_age_years', 'vehicle_age'],
    'average_charge_power_kw': ['average_charge_power_kw', 'charge_power_kw'],
    'regenerative_braking_usage': ['regenerative_braking_usage', 'regenerative_braking_efficiency'],
    'aggressive_acceleration_score': ['aggressive_acceleration_score', 'aggressive_acceleration_events'],
}


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
    - Temperatures: [-30°C, 80°C]
    - State of Health (SoH) and Health Percent: [0%, 100%]
    - Efficiencies & Ratios: [0.0, 1.0] or [0%, 100%]
    - Voltages: cell voltage [2.0V, 4.5V]
    """
    df_clean = df.copy()

    # Temperature columns
    temp_cols = [
        'cell_temperature_avg', 'cell_temperature_max', 'cell_temperature_min',
        'average_ambient_temperature', 'maximum_temperature', 'minimum_temperature'
    ]
    for col in temp_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].clip(lower=-30.0, upper=80.0)

    # Health & SOH columns
    soh_cols = ['battery_health_percent', 'battery_health_percentage', 'state_of_health']
    for col in soh_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].clip(lower=0.0, upper=100.0)

    # State of charge and Depth of discharge
    for col in ['state_of_charge', 'depth_of_discharge', 'capacity_loss_percent', 'capacity_loss_percentage']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].clip(lower=0.0, upper=100.0)

    # Charging ratios
    for col in ['fast_charge_ratio', 'slow_charge_ratio', 'overnight_charging_ratio', 'home_charging_ratio',
                 'highway_driving_ratio', 'city_driving_ratio']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].clip(lower=0.0, upper=1.0)

    # Internal resistance non-negative
    for col in ['internal_resistance', 'internal_resistance_mOhm']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].clip(lower=0.01, upper=5.0)

    return df_clean


def engineer_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives 8 Domain-Physics features + 4 Novel Electrochemical degradation features.

    Core 8 Domain-Physics Features:
    1. Arrhenius Aging Factor:
       np.exp(np.maximum(0, np.maximum(average_ambient_temp, cell_temperature_max) - 25) / 10.0)
    2. Cell Temperature Variance Delta:
       cell_temperature_max - cell_temperature_min
    3. Cell Voltage Spread:
       cell_voltage_max - cell_voltage_min
    4. Fast-Charge Cumulative Stress Index:
       fast_charge_ratio * cycle_count
    5. Resistance Growth Proxy:
       internal_resistance_mOhm * (1 + capacity_loss_percentage / 100.0)
    6. Equivalent Full Cycles (EFC) per 1,000 km:
       (cycle_count / np.maximum(odometer_km, 1.0)) * 1000
    7. Composite Physical Stress Aggregate:
       thermal_stress_score + charging_stress_score + driving_stress_score
    8. SoH Wear Loss per Cycle:
       (100 - battery_health_percentage) / np.maximum(cycle_count, 1)

    Novel Electrochemical Features:
    9. Thermal Dissipation Strain:
       (cell_temperature_max - ambient_temp) / np.maximum(cooling_system_health, 1.0)
    10. Internal Resistance per kWh Capacity:
        internal_resistance / np.maximum(battery_capacity_kwh, 1.0)
    11. Lithium Plating Susceptibility Index:
        np.maximum(0.0, 15.0 - ambient_temp) * fast_charge_ratio * (state_of_charge / 100.0)
    12. Degradation Acceleration Ratio:
        capacity_loss_percentage / np.maximum(vehicle_age_years, 0.1)
    """
    df_feat = df.copy()

    # Extract clean series with fallback resolution
    ambient_temp = get_series(df_feat, 'ambient_temp', default_val=25.0)
    cell_temp_max = get_series(df_feat, 'cell_temp_max', default_val=35.0)
    cell_temp_avg = get_series(df_feat, 'cell_temp_avg', default_val=30.0)
    
    # 2. Cell Temperature Variance Delta
    # If cell_temperature_min is present use it; otherwise infer from variance or symmetric delta
    if 'cell_temperature_min' in df_feat.columns:
        cell_temp_min = df_feat['cell_temperature_min']
    elif 'temperature_variance' in df_feat.columns and df_feat['temperature_variance'].notna().any():
        cell_temp_min = cell_temp_max - df_feat['temperature_variance'].fillna(5.0)
    else:
        cell_temp_min = cell_temp_avg - np.maximum(0.0, cell_temp_max - cell_temp_avg)

    # 1. Arrhenius Aging Factor:
    # Measures the exponential thermal acceleration of parasitic SEI layer growth
    effective_temp = np.maximum(ambient_temp.fillna(25.0), cell_temp_max.fillna(25.0))
    df_feat['arrhenius_aging_factor'] = np.exp(np.maximum(0.0, effective_temp - 25.0) / 10.0)

    # 2. Cell Temperature Variance Delta:
    df_feat['cell_temperature_variance_delta'] = cell_temp_max.fillna(30.0) - cell_temp_min.fillna(25.0)

    # 3. Cell Voltage Spread:
    if 'cell_voltage_max' in df_feat.columns and 'cell_voltage_min' in df_feat.columns:
        df_feat['cell_voltage_spread'] = df_feat['cell_voltage_max'] - df_feat['cell_voltage_min']
    elif 'voltage_imbalance' in df_feat.columns:
        # voltage_imbalance is in mV in typical BMS telematics, convert to V
        df_feat['cell_voltage_spread'] = df_feat['voltage_imbalance'] / 1000.0
    elif 'cell_voltage_std' in df_feat.columns:
        df_feat['cell_voltage_spread'] = df_feat['cell_voltage_std'] * 6.0
    else:
        df_feat['cell_voltage_spread'] = 0.020

    # 4. Fast-Charge Cumulative Stress Index:
    fast_charge_ratio = get_series(df_feat, 'fast_charge_ratio', default_val=0.3).fillna(0.3)
    cycle_count = get_series(df_feat, 'cycle_count', default_val=1000.0).fillna(1000.0)
    df_feat['fast_charge_cumulative_stress'] = fast_charge_ratio * cycle_count

    # 5. Resistance Growth Proxy:
    ir = get_series(df_feat, 'internal_resistance', default_val=0.4).fillna(0.4)
    cap_loss = get_series(df_feat, 'capacity_loss', default_val=15.0).fillna(15.0)
    df_feat['resistance_growth_proxy'] = ir * (1.0 + cap_loss / 100.0)

    # 6. Equivalent Full Cycles (EFC) per 1,000 km:
    odometer_km = get_series(df_feat, 'odometer_km', default_val=100000.0).fillna(100000.0)
    df_feat['efc_per_1000km'] = (cycle_count / np.maximum(odometer_km, 1.0)) * 1000.0

    # 7. Composite Physical Stress Aggregate:
    if 'thermal_stress_score' in df_feat.columns:
        thermal_stress = df_feat['thermal_stress_score'].fillna(30.0)
    else:
        th_score = get_series(df_feat, 'thermal_health_score', default_val=80.0).fillna(80.0)
        thermal_stress = 100.0 - th_score

    if 'charging_stress_score' in df_feat.columns:
        charging_stress = df_feat['charging_stress_score'].fillna(30.0)
    else:
        cq_score = get_series(df_feat, 'charging_quality_score', default_val=80.0).fillna(80.0)
        charging_stress = 100.0 - cq_score

    driving_stress = get_series(df_feat, 'driving_stress_score', default_val=40.0).fillna(40.0)
    df_feat['composite_physical_stress'] = thermal_stress + charging_stress + driving_stress

    # 8. SoH Wear Loss per Cycle:
    soh_pct = get_series(df_feat, 'battery_health_percent', default_val=85.0).fillna(85.0)
    df_feat['soh_wear_loss_per_cycle'] = (100.0 - soh_pct) / np.maximum(cycle_count, 1.0)

    # ================= NOVEL ELECTROCHEMICAL FEATURES =================
    # 9. Thermal Dissipation Strain:
    cooling_health = get_series(df_feat, 'cooling_system_health', default_val=80.0).fillna(80.0)
    df_feat['thermal_dissipation_strain'] = (
        (cell_temp_max.fillna(35.0) - ambient_temp.fillna(25.0)) / np.maximum(cooling_health, 1.0)
    )

    # 10. Internal Resistance per kWh Capacity:
    capacity_kwh = get_series(df_feat, 'battery_capacity_kwh', default_val=75.0).fillna(75.0)
    df_feat['ir_per_kwh'] = ir / np.maximum(capacity_kwh, 1.0)

    # 11. Lithium Plating Susceptibility Metric:
    soc = get_series(df_feat, 'state_of_charge', default_val=50.0).fillna(50.0)
    df_feat['lithium_plating_risk'] = (
        np.maximum(0.0, 15.0 - ambient_temp.fillna(25.0)) * fast_charge_ratio * (soc / 100.0)
    )

    # 12. Degradation Acceleration Index:
    vehicle_age = get_series(df_feat, 'vehicle_age_years', default_val=5.0).fillna(5.0)
    df_feat['degradation_acceleration_index'] = cap_loss / np.maximum(vehicle_age, 0.1)

    return df_feat
