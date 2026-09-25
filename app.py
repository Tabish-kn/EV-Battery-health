"""
Interactive Streamlit UI Dashboard for EV Battery Health & Failure Prediction.
Features:
- Real-Time Diagnostic Telemetry Input & Failure Probability Gauge
- Dynamic Physics-Engineered Metric Cards
- XGBoost Feature Importance & Electrochemical Degradation Analytics
- Fleet-Wide Telemetry Batch Processing Engine with Export
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from feature_engine import clip_sensor_spikes, engineer_physics_features

# ==========================================
# PAGE CONFIG & MODERN DARK THEME STYLING
# ==========================================
st.set_page_config(
    page_title="EV Battery Diagnostics & Failure Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Main container background */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.4);
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
    }
    .physics-tag {
        display: inline-block;
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 600;
        margin-left: 6px;
    }
    
    /* Risk Badges */
    .badge-low {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: #ecfdf5;
        padding: 8px 18px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    }
    .badge-moderate {
        background: linear-gradient(135deg, #92400e 0%, #b45309 100%);
        color: #fffbeb;
        padding: 8px 18px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.4);
    }
    .badge-high {
        background: linear-gradient(135deg, #991b1b 0%, #dc2626 100%);
        color: #fef2f2;
        padding: 8px 18px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.6);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 10px rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 25px rgba(239, 68, 68, 0.8); }
        100% { box-shadow: 0 0 10px rgba(239, 68, 68, 0.4); }
    }
    
    /* Tabs header */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #131b2e;
        padding: 8px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================
# ARTIFACT & MODEL CACHE LOADER
# ==========================================
@st.cache_resource
def load_pipeline_artifacts():
    bundle_file = "ev_pipeline_bundle.pkl"
    if not os.path.exists(bundle_file):
        st.error(f"Pipeline bundle '{bundle_file}' not found! Please run 'python eda_and_model.py' first.")
        st.stop()
    return joblib.load(bundle_file)


bundle = load_pipeline_artifacts()
model = bundle['model']
scaler = bundle['scaler']
num_imputer = bundle['num_imputer']
cat_imputer = bundle['cat_imputer']
encoder = bundle['encoder']
num_cols = bundle['num_cols']
cat_cols = bundle['cat_cols']
all_feature_names = bundle['all_feature_names']
physics_features = bundle['physics_features']
metrics = bundle['metrics']


def predict_single_pack(telemetry_dict):
    """Predicts battery failure probability for a single vehicle telemetry input."""
    df_raw = pd.DataFrame([telemetry_dict])
    df_clean = clip_sensor_spikes(df_raw)
    df_feat = engineer_physics_features(df_clean)

    # Transform numerical features
    X_num = num_imputer.transform(df_feat[num_cols])

    # Transform categorical features
    if cat_cols and encoder is not None:
        X_cat = cat_imputer.transform(df_feat[cat_cols])
        X_cat_enc = encoder.transform(X_cat)
        X_comb = np.hstack([X_num, X_cat_enc])
    else:
        X_comb = X_num

    X_scaled = scaler.transform(X_comb)
    proba = model.predict_proba(X_scaled)[0, 1]
    return proba, df_feat.iloc[0]


# ==========================================
# SIDEBAR: VEHICLE TELEMETRY CONTROLS
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/electric-vehicle.png", width=64)
    st.title("⚡ Powertrain Telemetry")
    st.caption("Adjust real-time battery pack sensor metrics:")

    with st.expander("🔋 Pack Architecture & Chemistry", expanded=True):
        chem = st.selectbox("Battery Chemistry", ["NMC", "LFP", "NCA", "LMO", "LTO"], index=0)
        drive_type = st.selectbox("Drive Configuration", ["AWD", "RWD", "FWD"], index=0)
        fleet_type = st.selectbox("Usage Sector", ["Fleet", "Private"], index=0)
        v_type = st.selectbox("Vehicle Class", ["Sedan", "SUV", "Crossover", "Truck", "Van", "Hatchback"], index=1)
        terrain = st.selectbox("Operating Terrain", ["Flat", "Coastal", "Desert", "Mountainous", "Hilly"], index=0)
        nominal_cap = st.slider("Nominal Capacity (kWh)", 25.0, 150.0, 82.0, 1.0)
        cycle_count = st.slider("Equivalent Cycle Count", 50, 4000, 1250, 25)
        odometer_km = st.number_input("Odometer (km)", value=145000.0, step=5000.0)
        vehicle_age = st.slider("Vehicle Age (Years)", 0.5, 15.0, 4.5, 0.1)

    with st.expander("🌡️ Thermal & Environmental", expanded=True):
        ambient_temp = st.slider("Ambient Temperature (°C)", -25.0, 50.0, 32.0, 1.0)
        cell_temp_max = st.slider("Cell Max Temperature (°C)", 15.0, 75.0, 42.0, 0.5)
        cell_temp_avg = st.slider("Cell Avg Temperature (°C)", 10.0, 60.0, 35.0, 0.5)
        cooling_health = st.slider("Cooling System Health (%)", 20.0, 100.0, 85.0, 1.0)
        thermal_health_score = st.slider("Thermal Health Score", 10.0, 100.0, 82.0, 1.0)

    with st.expander("⚡ Electrochemical & Voltage", expanded=False):
        soh_pct = st.slider("State of Health / SOH (%)", 40.0, 100.0, 82.0, 0.5)
        cap_loss = st.slider("Capacity Loss (%)", 0.0, 60.0, 18.0, 0.5)
        soc = st.slider("State of Charge (%)", 5.0, 100.0, 65.0, 1.0)
        internal_res = st.slider("Internal Resistance (mΩ)", 0.05, 1.50, 0.48, 0.01)
        voltage_imbalance = st.slider("Voltage Imbalance (mV)", 0.0, 60.0, 14.5, 0.5)
        cell_v_avg = st.slider("Cell Avg Voltage (V)", 3.2, 4.2, 3.75, 0.01)
        cell_v_std = st.slider("Cell Voltage Std (V)", 0.0005, 0.0500, 0.0120, 0.0010)
        pack_voltage = st.slider("Pack Voltage (V)", 250.0, 420.0, 380.0, 1.0)

    with st.expander("🔌 Charging & Driver Stress", expanded=False):
        fast_charge_ratio = st.slider("DC Fast-Charge Ratio", 0.0, 1.0, 0.45, 0.05)
        slow_charge_ratio = 1.0 - fast_charge_ratio
        avg_charge_kw = st.slider("Avg Charging Power (kW)", 7.0, 250.0, 75.0, 5.0)
        charging_qual_score = st.slider("Charging Quality Score", 10.0, 100.0, 85.0, 1.0)
        driving_stress = st.slider("Driving Stress Score", 10.0, 100.0, 45.0, 1.0)
        bms_warnings = st.slider("BMS Warning Count", 0, 15, 1, 1)
        prev_faults = st.slider("Previous Recorded Faults", 0, 10, 1, 1)
        thermal_runaway_risk = st.slider("Thermal Runaway Risk Index", 0.0, 80.0, 18.0, 1.0)

# Assemble telemetry input record
current_telemetry = {
    'battery_chemistry': chem,
    'drive_type': drive_type,
    'fleet_or_private': fleet_type,
    'vehicle_type': v_type,
    'terrain_type': terrain,
    'battery_capacity_kwh': nominal_cap,
    'cycle_count': cycle_count,
    'odometer_km': odometer_km,
    'vehicle_age_years': vehicle_age,
    'average_ambient_temperature': ambient_temp,
    'cell_temperature_max': cell_temp_max,
    'cell_temperature_avg': cell_temp_avg,
    'minimum_temperature': ambient_temp - 8.0,
    'maximum_temperature': ambient_temp + 8.0,
    'cooling_system_health': cooling_health,
    'thermal_health_score': thermal_health_score,
    'battery_health_percent': soh_pct,
    'state_of_health': soh_pct,
    'capacity_loss_percent': cap_loss,
    'remaining_capacity': nominal_cap * (soh_pct / 100.0),
    'state_of_charge': soc,
    'depth_of_discharge': 100.0 - soc,
    'internal_resistance': internal_res,
    'voltage_imbalance': voltage_imbalance,
    'cell_voltage_avg': cell_v_avg,
    'cell_voltage_std': cell_v_std,
    'pack_voltage': pack_voltage,
    'fast_charge_ratio': fast_charge_ratio,
    'slow_charge_ratio': slow_charge_ratio,
    'average_charge_power_kw': avg_charge_kw,
    'charging_quality_score': charging_qual_score,
    'driving_stress_score': driving_stress,
    'BMS_warning_count': bms_warnings,
    'previous_faults': prev_faults,
    'thermal_runaway_risk': thermal_runaway_risk,
    'charge_efficiency': 95.5,
    'discharge_efficiency': 94.0,
    'charging_cycles_last_month': 25,
    'average_charging_time': 120.0,
    'overnight_charging_ratio': 0.40,
    'home_charging_ratio': 0.60,
    'charging_interruptions': 1,
    'overcharge_events': 0,
    'average_speed': 55.0,
    'average_trip_distance': 45.0,
    'aggressive_acceleration_score': 35.0,
    'hard_braking_score': 30.0,
    'regenerative_braking_usage': 60.0,
    'highway_driving_ratio': 0.45,
    'city_driving_ratio': 0.55,
    'daily_distance': 70.0,
    'humidity': 50.0,
    'altitude': 250.0,
    'dust_exposure': 35.0,
    'last_service_days': 120.0,
    'firmware_updates': 8,
    'maintenance_score': 85.0,
    'temperature_variance': cell_temp_max - cell_temp_avg,
    'sensor_fault_count': 0,
    'abnormal_voltage_events': 0,
    'battery_stress_index': 45.0,
    'aging_score': 55.0,
    'predicted_remaining_life_cycles': max(200, int(6000 - cycle_count * 1.8)),
    'manufacturing_year': int(2026 - vehicle_age)
}


# ==========================================
# APP HEADER
# ==========================================
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #1e293b; padding-bottom: 15px; margin-bottom: 25px;">
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; color: #f8fafc; font-weight: 800;">
                ⚡ EV Battery Health & Powertrain Diagnostics
            </h1>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 1.0rem;">
                Physics-Informed XGBoost Machine Learning System for Battery Pack Failure Prediction
            </p>
        </div>
        <div style="text-align: right;">
            <span style="background: #1e3a8a; color: #93c5fd; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">ROC-AUC: 0.980</span>
            <span style="background: #065f46; color: #a7f3d0; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; margin-left: 6px;">PR-AUC: 0.758</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "🎛️ Real-Time Diagnostic Dashboard",
    "📊 Feature Importance & Analytics",
    "🚛 Fleet Batch Processing Engine"
])


# =========================================================
# TAB 1: REAL-TIME DIAGNOSTIC DASHBOARD
# =========================================================
with tab1:
    prob, feat_row = predict_single_pack(current_telemetry)
    prob_pct = prob * 100.0

    col_gauge, col_status = st.columns([1.2, 1.8])

    with col_gauge:
        st.subheader("Predicted Failure Probability")
        
        # Plotly Gauge Chart
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Battery Failure Risk", 'font': {'size': 20, 'color': '#e2e8f0'}},
            number={'suffix': "%", 'font': {'size': 38, 'color': '#ffffff'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
                'bar': {'color': "#ef4444" if prob_pct > 55 else ("#f59e0b" if prob_pct > 20 else "#10b981")},
                'bgcolor': "#1e293b",
                'borderwidth': 2,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 20], 'color': 'rgba(16, 185, 129, 0.15)'},
                    {'range': [20, 55], 'color': 'rgba(245, 158, 11, 0.15)'},
                    {'range': [55, 100], 'color': 'rgba(239, 68, 68, 0.25)'}
                ],
                'threshold': {
                    'line': {'color': "#ef4444", 'width': 4},
                    'thickness': 0.8,
                    'value': 55.0
                }
            }
        ))
        gauge_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#e2e8f0"},
            height=290,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(gauge_fig, width='stretch')

    with col_status:
        st.subheader("Diagnostic Status & BMS Intervention")
        if prob_pct < 20:
            st.markdown(
                f"""
                <div class="metric-card" style="border-left: 6px solid #10b981;">
                    <span class="badge-low">🟢 LOW RISK ({prob_pct:.1f}%)</span>
                    <h3 style="color: #ecfdf5; margin-top: 14px;">Pack Operating Within Nominal Envelope</h3>
                    <p style="color: #94a3b8; font-size: 0.95rem;">
                        Electrochemical degradation rate is nominal. SEI layer growth and internal resistance
                        remain well within allowable automotive tolerance limits. Continue standard maintenance schedule.
                    </p>
                    <ul style="color: #cbd5e1; font-size: 0.88rem; padding-left: 20px;">
                        <li>Cell temperature variance is controlled: <b>{feat_row['cell_temperature_variance_delta']:.1f}°C</b></li>
                        <li>Arrhenius thermal multiplier: <b>{feat_row['arrhenius_aging_factor']:.2f}x</b> nominal baseline</li>
                        <li>No proactive de-rating required.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif prob_pct <= 55:
            st.markdown(
                f"""
                <div class="metric-card" style="border-left: 6px solid #f59e0b;">
                    <span class="badge-moderate">🟡 MODERATE RISK ({prob_pct:.1f}%)</span>
                    <h3 style="color: #fffbeb; margin-top: 14px;">Early Degradation & Stress Symptoms Detected</h3>
                    <p style="color: #94a3b8; font-size: 0.95rem;">
                        The battery pack exhibits elevated stress indicators. Resistance growth or cumulative fast-charge
                        cycling is accelerating cell divergence. Recommend preventative service window.
                    </p>
                    <ul style="color: #cbd5e1; font-size: 0.88rem; padding-left: 20px;">
                        <li>Initiate pack cell balancing cycle to mitigate voltage spread (<b>{feat_row['cell_voltage_spread']*1000:.1f} mV</b>).</li>
                        <li>Advise driver to cap fast charging power and avoid low-ambient rapid charging.</li>
                        <li>Inspect cooling system loop efficiency and thermal interface material (TIM).</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="metric-card" style="border-left: 6px solid #ef4444;">
                    <span class="badge-high">🔴 HIGH / CRITICAL RISK ({prob_pct:.1f}%)</span>
                    <h3 style="color: #fef2f2; margin-top: 14px;">Critical Failure Imminent - Immediate Intervention</h3>
                    <p style="color: #fca5a5; font-size: 0.95rem;">
                        Severe electrochemical degradation detected. High internal resistance growth, severe Arrhenius thermal
                        stress, or elevated thermal runaway risk indices flag an impending cell or module-level breakdown.
                    </p>
                    <ul style="color: #cbd5e1; font-size: 0.88rem; padding-left: 20px;">
                        <li><b>Immediate Action:</b> Restrict DC fast charging (cap at 0.5C max).</li>
                        <li><b>Safety Action:</b> Trigger BMS warning DTC code; schedule physical pack bench inspection.</li>
                        <li><b>Thermal Mitigation:</b> Force active coolant pre-chilling upon peak thermal load.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### 🔬 Engineered Domain-Physics Features")
    st.caption("Derived from physical laws of battery electrochemistry, Arrhenius kinetics, and thermodynamics:")

    row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)
    with row1_c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">1. Arrhenius Aging Factor <span class="physics-tag">PHYSICS</span></div>
                <div class="metric-value">{feat_row['arrhenius_aging_factor']:.2f}x</div>
                <div class="metric-sub">exp((max(T)-25)/10) · SEI reaction rate</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with row1_c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">2. Cell Temp Delta <span class="physics-tag">PHYSICS</span></div>
                <div class="metric-value">{feat_row['cell_temperature_variance_delta']:.2f} °C</div>
                <div class="metric-sub">T_max - T_min · Thermal gradient</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with row1_c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">3. Cell Voltage Spread <span class="physics-tag">PHYSICS</span></div>
                <div class="metric-value">{feat_row['cell_voltage_spread']*1000:.1f} mV</div>
                <div class="metric-sub">V_max - V_min · Cell imbalance</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with row1_c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">4. Fast-Charge Stress <span class="physics-tag">PHYSICS</span></div>
                <div class="metric-value">{feat_row['fast_charge_cumulative_stress']:.0f} cyc</div>
                <div class="metric-sub">Fast_ratio × Total cycles</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    row2_c1, row2_c2, row2_c3, row2_c4 = st.columns(4)
    with row2_c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">5. Resistance Growth <span class="physics-tag">PHYSICS</span></div>
                <div class="metric-value">{feat_row['resistance_growth_proxy']:.3f} mΩ</div>
                <div class="metric-sub">R_int × (1 + Cap_loss/100)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with row2_c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">6. EFC per 1,000 km <span class="physics-tag">PHYSICS</span></div>
                <div class="metric-value">{feat_row['efc_per_1000km']:.2f}</div>
                <div class="metric-sub">(Cycles / Odometer) × 1000</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with row2_c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">7. Composite Stress <span class="physics-tag">PHYSICS</span></div>
                <div class="metric-value">{feat_row['composite_physical_stress']:.1f} pts</div>
                <div class="metric-sub">Thermal + Charge + Drive Stress</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with row2_c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">8. SoH Wear Rate <span class="physics-tag">PHYSICS</span></div>
                <div class="metric-value">{feat_row['soh_wear_loss_per_cycle']*100:.3f}%</div>
                <div class="metric-sub">ΔHealth / Cycle count</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Additional Novel Features Row
    st.markdown("#### 🌟 Novel Powertrain & Electrochemical Diagnostics")
    nov_c1, nov_c2, nov_c3, nov_c4 = st.columns(4)
    with nov_c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Thermal Dissipation Strain</div>
                <div class="metric-value">{feat_row['thermal_dissipation_strain']:.3f}</div>
                <div class="metric-sub">ΔT_ambient / Cooling_health</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with nov_c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">IR Density per kWh</div>
                <div class="metric-value">{feat_row['ir_per_kwh']:.4f}</div>
                <div class="metric-sub">mΩ per Nominal kWh</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with nov_c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Lithium Plating Risk</div>
                <div class="metric-value">{feat_row['lithium_plating_risk']:.2f}</div>
                <div class="metric-sub">Low Temp × Fast Charge × SOC</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with nov_c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Degradation Acceleration</div>
                <div class="metric-value">{feat_row['degradation_acceleration_index']:.2f} %/yr</div>
                <div class="metric-sub">Capacity Loss / Vehicle Age</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# TAB 2: FEATURE IMPORTANCE & ANALYTICS
# =========================================================
with tab2:
    st.subheader("Model Explainability & Feature Attribution")
    st.write(
        "Evaluation of global feature weights from the production XGBoost model (`scale_pos_weight = 13.45`). "
        "Engineered physics metrics provide direct thermodynamic grounding, helping the tree models detect "
        "nonlinear interactions."
    )

    fi = model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': fi,
        'Category': ['Physics-Engineered' if f in physics_features else 'Raw Telemetry' for f in all_feature_names]
    }).sort_values('Importance', ascending=False)

    col_fi, col_meta = st.columns([2.2, 1.0])

    with col_fi:
        top_25 = fi_df.head(20).iloc[::-1]
        fig_bar = px.bar(
            top_25,
            x='Importance',
            y='Feature',
            color='Category',
            color_discrete_map={'Physics-Engineered': '#ef4444', 'Raw Telemetry': '#3b82f6'},
            orientation='h',
            title="Top 20 XGBoost Feature Importances (Gain Ratio)",
            labels={'Importance': 'Relative Importance Weight', 'Feature': ''}
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#cbd5e1'},
            height=580,
            margin=dict(l=10, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_bar.update_xaxes(showgrid=True, gridcolor='#1e293b')
        st.plotly_chart(fig_bar, width='stretch')

    with col_meta:
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top: 35px;">
                <h4 style="margin: 0 0 10px 0; color: #60a5fa;">Validation Benchmark</h4>
                <p style="color: #94a3b8; font-size: 0.85rem;">Trained on 16,000 packs, evaluated on 4,000 unseen test packs:</p>
                <div style="margin: 8px 0;"><b>ROC-AUC:</b> <span style="color: #34d399;">{metrics['roc_auc']:.4f}</span></div>
                <div style="margin: 8px 0;"><b>PR-AUC:</b> <span style="color: #34d399;">{metrics['pr_auc']:.4f}</span></div>
                <div style="margin: 8px 0;"><b>Base Failure Rate:</b> <span style="color: #f87171;">{metrics['base_fail_rate']:.2%}</span></div>
                <div style="margin: 8px 0;"><b>Failure Recall:</b> <span style="color: #38bdf8;">85.2%</span></div>
                <div style="margin: 8px 0;"><b>Overall Accuracy:</b> <span style="color: #38bdf8;">95.3%</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="metric-card">
                <h4 style="margin: 0 0 10px 0; color: #f87171;">Physics Engine Impact</h4>
                <p style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.4;">
                    <b>Resistance Growth Proxy</b> and <b>Arrhenius Aging Factor</b> directly isolate nonlinear
                    temperature-dependent SEI accumulation and internal Joule heating, outperforming generic raw sensor inputs.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Radar Chart of Stress Factors for Current Telemetry
    st.markdown("### 🕸️ Multi-Dimensional Stress Profiler")
    radar_categories = [
        'Arrhenius Thermal',
        'Temp Variance Delta',
        'Voltage Spread',
        'Fast Charge Stress',
        'Resistance Growth',
        'Composite Stress'
    ]
    # Normalize 0 to 100 for radar visual
    r_arrhenius = min(100.0, feat_row['arrhenius_aging_factor'] * 10.0)
    r_temp_var = min(100.0, feat_row['cell_temperature_variance_delta'] * 2.5)
    r_volt_spread = min(100.0, feat_row['cell_voltage_spread'] * 2000.0)
    r_fast_charge = min(100.0, (feat_row['fast_charge_cumulative_stress'] / 1000.0) * 100.0)
    r_res_growth = min(100.0, feat_row['resistance_growth_proxy'] * 100.0)
    r_composite = min(100.0, feat_row['composite_physical_stress'])

    radar_values = [r_arrhenius, r_temp_var, r_volt_spread, r_fast_charge, r_res_growth, r_composite]
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=radar_values + [radar_values[0]],
        theta=radar_categories + [radar_categories[0]],
        fill='toself',
        fillcolor='rgba(239, 68, 68, 0.3)',
        line=dict(color='#ef4444', width=2),
        name="Current Vehicle Stress Profile"
    ))
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#64748b")),
            bgcolor="rgba(15, 23, 42, 0.8)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        height=400,
        margin=dict(l=40, r=40, t=30, b=30)
    )
    st.plotly_chart(radar_fig, width='stretch')


# =========================================================
# TAB 3: FLEET BATCH PROCESSING ENGINE
# =========================================================
with tab3:
    st.subheader("Fleet-Wide Telemetry Batch Inference")
    st.write(
        "Upload a fleet CSV telemetry export or run automated diagnostics on a preloaded fleet subset. "
        "The physics engine calculates all 8 domain features and flags high-risk vehicles for maintenance dispatch."
    )

    batch_col1, batch_col2 = st.columns([2, 1])

    with batch_col1:
        uploaded_file = st.file_uploader("Upload Fleet Telemetry CSV", type=["csv"])
    with batch_col2:
        st.write("Or load quick benchmark fleet batch:")
        use_sample = st.button("🚀 Load 1,000 Vehicle Fleet Telemetry Sample", width='stretch')

    fleet_df = None
    if uploaded_file is not None:
        fleet_df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded {len(fleet_df):,} vehicles successfully!")
    elif use_sample:
        if os.path.exists("ev_battery_health.csv"):
            fleet_df = pd.read_csv("ev_battery_health.csv", nrows=1000)
            st.info("Loaded 1,000 vehicle fleet telemetry subset.")
        else:
            st.error("Dataset 'ev_battery_health.csv' not found.")

    if fleet_df is not None:
        with st.spinner("Executing Physics Feature Transformation & XGBoost Inference..."):
            # Clean and feature engineer
            clean_fleet = clip_sensor_spikes(fleet_df)
            feat_fleet = engineer_physics_features(clean_fleet)

            # Preprocess
            X_batch_num = num_imputer.transform(feat_fleet[num_cols])
            if cat_cols and encoder is not None:
                X_batch_cat = cat_imputer.transform(feat_fleet[cat_cols])
                X_batch_cat_enc = encoder.transform(X_batch_cat)
                X_batch_comb = np.hstack([X_batch_num, X_batch_cat_enc])
            else:
                X_batch_comb = X_batch_num

            X_batch_scaled = scaler.transform(X_batch_comb)
            batch_probas = model.predict_proba(X_batch_scaled)[:, 1]

            # Results dataframe
            results_df = fleet_df.copy()
            results_df['failure_probability'] = np.round(batch_probas, 4)
            results_df['risk_category'] = pd.cut(
                results_df['failure_probability'],
                bins=[-0.01, 0.20, 0.55, 1.01],
                labels=['Low Risk', 'Moderate Risk', 'High Risk']
            )

            # Add core physics features into results
            for pf in physics_features:
                results_df[pf] = np.round(feat_fleet[pf], 4)

        # Summary KPIs
        total_veh = len(results_df)
        low_count = (results_df['risk_category'] == 'Low Risk').sum()
        mod_count = (results_df['risk_category'] == 'Moderate Risk').sum()
        high_count = (results_df['risk_category'] == 'High Risk').sum()

        st.markdown("### 📈 Fleet Risk Summary")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Total Evaluated Packs</div>
                    <div class="metric-value">{total_veh:,}</div>
                    <div class="metric-sub">Active fleet units</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with kpi2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Nominal / Low Risk</div>
                    <div class="metric-value" style="color: #34d399;">{low_count:,} ({low_count/total_veh:.1%})</div>
                    <div class="metric-sub">Healthy operating envelope</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with kpi3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Moderate Risk (Monitor)</div>
                    <div class="metric-value" style="color: #fbbf24;">{mod_count:,} ({mod_count/total_veh:.1%})</div>
                    <div class="metric-sub">Early degradation flags</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with kpi4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">High Risk (Immediate Dispatch)</div>
                    <div class="metric-value" style="color: #f87171;">{high_count:,} ({high_count/total_veh:.1%})</div>
                    <div class="metric-sub">Critical failure probability > 55%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Charts: Donut + Histogram
        ch_col1, ch_col2 = st.columns([1, 1.5])
        with ch_col1:
            donut_fig = px.pie(
                results_df,
                names='risk_category',
                title="Fleet Risk Distribution",
                color='risk_category',
                color_discrete_map={'Low Risk': '#10b981', 'Moderate Risk': '#f59e0b', 'High Risk': '#ef4444'},
                hole=0.45
            )
            donut_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#cbd5e1'},
                height=320,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(donut_fig, width='stretch')

        with ch_col2:
            hist_fig = px.histogram(
                results_df,
                x='failure_probability',
                color='risk_category',
                nbins=40,
                title="Failure Probability Distribution Across Fleet",
                color_discrete_map={'Low Risk': '#10b981', 'Moderate Risk': '#f59e0b', 'High Risk': '#ef4444'},
                labels={'failure_probability': 'Predicted Failure Probability'}
            )
            hist_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#cbd5e1'},
                height=320,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(hist_fig, width='stretch')

        # Filter and Display Table
        st.markdown("### 📋 Fleet Telemetry & Risk Explorer")
        risk_filter = st.multiselect(
            "Filter by Risk Category:",
            ['High Risk', 'Moderate Risk', 'Low Risk'],
            default=['High Risk', 'Moderate Risk']
        )
        display_df = results_df[results_df['risk_category'].isin(risk_filter)]

        display_cols = [
            c for c in [
                'vehicle_id', 'vehicle_brand', 'vehicle_model', 'battery_chemistry',
                'failure_probability', 'risk_category', 'battery_health_percent',
                'arrhenius_aging_factor', 'cell_temperature_max', 'fast_charge_cumulative_stress',
                'resistance_growth_proxy', 'cell_voltage_spread', 'odometer_km'
            ] if c in display_df.columns
        ]

        st.dataframe(
            display_df[display_cols].sort_values('failure_probability', ascending=False),
            width='stretch',
            height=320
        )

        # Download button for high risk units
        high_risk_export = results_df[results_df['risk_category'] == 'High Risk']
        csv_bytes = high_risk_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download High-Risk Telemetry Report ({len(high_risk_export)} Vehicles CSV)",
            data=csv_bytes,
            file_name="high_risk_ev_battery_telemetry.csv",
            mime="text/csv"
        )
