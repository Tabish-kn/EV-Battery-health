"""
Configuration and constants for EV Battery Intelligence Platform.
Defines paths, feature aliases, physical telemetry boundaries, model hyperparameters,
and risk scoring weights.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "ev_battery_health.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "ev_battery_health_featured.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample" / "sample_battery_telemetry.csv"

MODELS_DIR = PROJECT_ROOT / "models"
FAILURE_MODEL_DIR = MODELS_DIR / "failure_model"
SOH_MODEL_DIR = MODELS_DIR / "soh_model"
RUL_MODEL_DIR = MODELS_DIR / "rul_model"
ANOMALY_MODEL_DIR = MODELS_DIR / "anomaly_model"
PREPROCESSORS_DIR = MODELS_DIR / "preprocessors"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
GENERATED_REPORTS_DIR = REPORTS_DIR / "generated"

# Ensure runtime directories exist
for path in [
    DATA_DIR / "raw", DATA_DIR / "processed", DATA_DIR / "sample",
    FAILURE_MODEL_DIR, SOH_MODEL_DIR, RUL_MODEL_DIR, ANOMALY_MODEL_DIR, PREPROCESSORS_DIR,
    FIGURES_DIR, GENERATED_REPORTS_DIR
]:
    os.makedirs(path, exist_ok=True)

# Random Seed for reproducibility
RANDOM_STATE = 42

# Target Column Names
TARGET_FAILURE = "battery_failure"
TARGET_SOH = "state_of_health"
TARGET_SOH_ALT = "battery_health_percent"
TARGET_RUL = "predicted_remaining_life_cycles"

# Identifier & Metadata Columns (Excluded from model features)
ID_COLUMNS = ["vehicle_id", "battery_serial"]

# Potential Categorical Columns
CATEGORICAL_COLUMNS = [
    "vehicle_brand",
    "vehicle_model",
    "vehicle_type",
    "battery_manufacturer",
    "battery_chemistry",
    "drive_type",
    "fleet_or_private",
    "terrain_type",
]

# Physical Telemetry Realistic Boundaries for Sensor Spike Clipping & Validation
PHYSICAL_BOUNDS = {
    "pack_voltage": (100.0, 900.0),             # Pack Voltage in Volts
    "cell_voltage_avg": (2.0, 4.5),              # Nominal cell voltage in Volts
    "cell_voltage_min": (1.8, 4.3),
    "cell_voltage_max": (2.2, 4.6),
    "cell_voltage_std": (0.0, 0.20),
    "voltage_imbalance": (0.0, 200.0),           # Imbalance in mV
    "cell_temperature_avg": (-30.0, 85.0),       # Temperature in Celsius
    "cell_temperature_max": (-30.0, 95.0),
    "cell_temperature_min": (-40.0, 75.0),
    "average_ambient_temperature": (-40.0, 60.0),
    "maximum_temperature": (-30.0, 75.0),
    "minimum_temperature": (-45.0, 50.0),
    "internal_resistance": (0.01, 5.0),          # in mOhm
    "battery_capacity_kwh": (10.0, 250.0),       # Pack capacity in kWh
    "state_of_charge": (0.0, 100.0),             # SOC %
    "depth_of_discharge": (0.0, 100.0),          # DOD %
    "state_of_health": (0.0, 100.0),             # SOH %
    "battery_health_percent": (0.0, 100.0),
    "charge_efficiency": (50.0, 100.0),
    "discharge_efficiency": (50.0, 100.0),
    "capacity_loss_percent": (0.0, 100.0),
    "fast_charge_ratio": (0.0, 1.0),
    "slow_charge_ratio": (0.0, 1.0),
    "overnight_charging_ratio": (0.0, 1.0),
    "home_charging_ratio": (0.0, 1.0),
    "highway_driving_ratio": (0.0, 1.0),
    "city_driving_ratio": (0.0, 1.0),
    "cooling_system_health": (0.0, 100.0),
    "thermal_runaway_risk": (0.0, 100.0),
    "battery_stress_index": (0.0, 100.0),
    "aging_score": (0.0, 100.0),
    "thermal_health_score": (0.0, 100.0),
    "charging_quality_score": (0.0, 100.0),
    "driving_stress_score": (0.0, 100.0),
    "maintenance_score": (0.0, 100.0),
    "odometer_km": (0.0, 1_000_000.0),
    "cycle_count": (0.0, 20_000.0),
    "vehicle_age_years": (0.0, 30.0),
}

# Column Aliases for Flexible CSV Ingestion
COLUMN_ALIASES = {
    "ambient_temp": ["average_ambient_temperature", "average_ambient_temp", "ambient_temperature"],
    "ambient_temp_max": ["maximum_temperature", "maximum_ambient_temp"],
    "ambient_temp_min": ["minimum_temperature", "minimum_ambient_temp"],
    "cell_temp_max": ["cell_temperature_max", "cell_temp_max"],
    "cell_temp_avg": ["cell_temperature_avg", "cell_temp_avg"],
    "cell_temp_min": ["cell_temperature_min", "cell_temp_min"],
    "temp_variance": ["temperature_variance", "cell_temp_variance"],
    "cell_voltage_avg": ["cell_voltage_avg", "cell_voltage_mean"],
    "cell_voltage_std": ["cell_voltage_std", "cell_voltage_sigma"],
    "cell_voltage_max": ["cell_voltage_max"],
    "cell_voltage_min": ["cell_voltage_min"],
    "voltage_imbalance": ["voltage_imbalance", "cell_voltage_imbalance"],
    "pack_voltage": ["pack_voltage"],
    "cycle_count": ["cycle_count", "charging_cycles"],
    "fast_charge_ratio": ["fast_charge_ratio"],
    "slow_charge_ratio": ["slow_charge_ratio"],
    "internal_resistance": ["internal_resistance", "internal_resistance_mOhm", "battery_internal_resistance"],
    "capacity_loss": ["capacity_loss_percent", "capacity_loss_percentage"],
    "soh": ["state_of_health", "battery_health_percent", "battery_health_percentage"],
    "battery_health_percent": ["battery_health_percent", "battery_health_percentage", "state_of_health"],
    "odometer_km": ["odometer_km", "odometer"],
    "thermal_health_score": ["thermal_health_score", "thermal_stress_score"],
    "charging_quality_score": ["charging_quality_score", "charging_stress_score"],
    "driving_stress_score": ["driving_stress_score"],
    "cooling_system_health": ["cooling_system_health"],
    "battery_capacity_kwh": ["battery_capacity_kwh", "nominal_capacity_kwh"],
    "state_of_charge": ["state_of_charge", "soc"],
    "vehicle_age_years": ["vehicle_age_years", "vehicle_age"],
    "average_charge_power_kw": ["average_charge_power_kw", "charge_power_kw"],
    "regenerative_braking_usage": ["regenerative_braking_usage", "regenerative_braking_efficiency"],
    "aggressive_acceleration_score": ["aggressive_acceleration_score", "aggressive_acceleration_events"],
}

# Risk Scoring Weight Configuration
RISK_WEIGHTS = {
    "failure_prob": 0.30,
    "thermal_risk": 0.20,
    "electrical_risk": 0.15,
    "charging_stress": 0.15,
    "degradation_risk": 0.10,
    "anomaly_risk": 0.10,
}
