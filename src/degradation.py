"""
Battery Electrochemical Degradation & Aging Kinetics Module.
Implements empirical and semi-empirical degradation models including Arrhenius thermal aging,
Solid Electrolyte Interphase (SEI) growth kinetics, capacity fade trajectories,
and internal resistance growth curves.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def calculate_arrhenius_acceleration(
    temperature_celsius: float,
    reference_temp_celsius: float = 25.0,
    activation_energy_ev: float = 0.35
) -> float:
    """
    Calculates thermal degradation acceleration factor using the Arrhenius relationship:
    k(T) / k(T_ref) = exp( - (E_a / k_B) * (1/T - 1/T_ref) )
    
    Args:
        temperature_celsius: Operating cell or ambient temperature.
        reference_temp_celsius: Nominal benchmark temperature (25°C).
        activation_energy_ev: Effective activation energy for SEI growth (~0.35 - 0.40 eV).
        
    Returns:
        Acceleration factor (> 1.0 indicates accelerated aging).
    """
    k_B = 8.617333262145e-5  # Boltzmann constant in eV/K
    T = max(temperature_celsius + 273.15, 230.0)
    T_ref = reference_temp_celsius + 273.15
    
    exponent = -(activation_energy_ev / k_B) * (1.0 / T - 1.0 / T_ref)
    factor = float(np.exp(np.clip(exponent, -5.0, 5.0)))
    return max(0.1, factor)


def simulate_degradation_trajectory(
    current_soh: float,
    current_cycle: int,
    fast_charge_ratio: float = 0.3,
    avg_temperature_c: float = 30.0,
    project_cycles: int = 4000,
    eol_soh_threshold: float = 70.0
) -> Dict[str, Any]:
    """
    Simulates longitudinal SOH degradation curve from current state to projected cycles
    incorporating empirical square-root cycle aging (SEI growth) and thermal acceleration.
    
    Model:
    Delta_SOH(n) = beta_0 * Arrhenius(T) * (1 + 0.8 * fast_charge_ratio) * sqrt(n)
    """
    arrhenius_factor = calculate_arrhenius_acceleration(avg_temperature_c)
    stress_multiplier = arrhenius_factor * (1.0 + 0.75 * fast_charge_ratio)
    
    # Calibrated nominal degradation slope per sqrt(cycle)
    base_fade_rate = 0.35 * stress_multiplier
    
    cycles = np.linspace(current_cycle, current_cycle + project_cycles, 100)
    delta_cycles = cycles - current_cycle
    
    # Capacity fade with non-linear transition near EOL
    soh_curve = current_soh - (base_fade_rate * np.sqrt(np.maximum(0, delta_cycles)))
    
    # Add non-linear acceleration when SOH drops below 75% (loss of active material)
    kneepoint_mask = soh_curve < 75.0
    if np.any(kneepoint_mask):
        excess_fade = (75.0 - soh_curve[kneepoint_mask]) ** 1.15 * 0.15
        soh_curve[kneepoint_mask] -= excess_fade
        
    soh_curve = np.clip(soh_curve, 0.0, 100.0)
    
    # Estimate cycles to EOL threshold
    eol_indices = np.where(soh_curve <= eol_soh_threshold)[0]
    if len(eol_indices) > 0:
        cycles_to_eol = float(cycles[eol_indices[0]] - current_cycle)
    else:
        # Extrapolate
        remaining_soh_drop = max(0.0, current_soh - eol_soh_threshold)
        cycles_to_eol = float((remaining_soh_drop / max(base_fade_rate, 1e-4)) ** 2)

    return {
        "cycles": cycles.tolist(),
        "projected_soh": [round(float(s), 2) for s in soh_curve],
        "arrhenius_factor": round(arrhenius_factor, 2),
        "stress_multiplier": round(stress_multiplier, 2),
        "estimated_cycles_to_eol": round(cycles_to_eol, 0),
        "eol_threshold": eol_soh_threshold
    }


def analyze_cohort_degradation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes historical degradation trends across battery chemistries and drive types.
    """
    analysis = {}
    
    if "battery_chemistry" in df.columns and "state_of_health" in df.columns:
        chem_summary = df.groupby("battery_chemistry").agg(
            mean_soh=("state_of_health", "mean"),
            median_soh=("state_of_health", "median"),
            mean_cycles=("cycle_count", "mean"),
            failure_rate=("battery_failure", "mean") if "battery_failure" in df.columns else ("state_of_health", "count")
        ).round(2).to_dict(orient="index")
        analysis["chemistry_degradation"] = chem_summary

    if "vehicle_age_years" in df.columns and "state_of_health" in df.columns:
        valid = df[["vehicle_age_years", "state_of_health"]].dropna()
        if len(valid) > 10:
            slope, intercept = np.polyfit(valid["vehicle_age_years"], valid["state_of_health"], 1)
            analysis["annual_fade_rate_percent"] = round(float(-slope), 2)
            
    return analysis
