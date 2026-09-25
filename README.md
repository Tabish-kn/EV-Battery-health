# ⚡ Physics-Informed EV Battery Failure Prediction & Telematics Diagnostics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20v3.4-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![ROC-AUC](https://img.shields.io/badge/Test%20ROC--AUC-0.9797-brightgreen.svg)](#-model-performance--evaluation)
[![PR-AUC](https://img.shields.io/badge/Test%20PR--AUC-0.7585-brightgreen.svg)](#-model-performance--evaluation)

Production-grade Machine Learning pipeline and interactive telemetry diagnostic dashboard designed for Electric Vehicle (EV) battery pack failure prediction (`battery_failure`: 0 = Healthy, 1 = Failure) across 20,000 commercial and passenger fleet battery records.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    A[Raw Telemetry Stream / CSV\n68 Telematics Sensors] --> B[Sensor Spike Cleaner\nPhysical Bounds: [-30°C, 80°C], [0%, 100%]]
    B --> C[Physics Feature Engine\n8 Core Laws + 4 Novel Electrochemical Metrics]
    C --> D[Preprocessing Pipeline\nMedian Imputer + One-Hot Encoder + StandardScaler]
    D --> E[XGBoost Classifier\nClass Imbalance: scale_pos_weight=13.45]
    E --> F[Serialized Artifacts\nmodel.pkl, scaler.pkl, bundle.pkl]
    F --> G[Interactive Streamlit Dashboard\napp.py]
    G --> H1[Tab 1: Real-Time Diagnostic Gauge]
    G --> H2[Tab 2: Feature Importance & Analytics]
    G --> H3[Tab 3: Fleet Batch Processing Engine]
```

---

## 🔬 Electrochemical & Domain-Physics Feature Rationale

Generic statistical models often struggle with sensor noise and non-linear interactions during battery degradation. This system implements **8 core domain-physics features** and **4 novel electrochemical degradation metrics** directly derived from physical principles:

### 1. Arrhenius Aging Factor

$$\text{Arrhenius Aging Factor} = \exp\left( \frac{\max\left(0, \max(T_{\text{ambient}}, T_{\text{cell, max}}) - 25\right)}{10.0} \right)$$

- **Physical Justification:** The rate of parasitic chemical reactions at the anode—principally Solid Electrolyte Interphase (SEI) growth and active lithium consumption—follows the Arrhenius rate law. Reaction kinetics approximately double for every 10°C rise above nominal reference temperature ($25^\circ\text{C}$). Cells operating under elevated peak thermal conditions degrade exponentially faster.

### 2. Cell Temperature Variance Delta

$$\Delta T_{\text{cell}} = T_{\text{cell, max}} - T_{\text{cell, min}}$$

- **Physical Justification:** Battery packs consist of hundreds of tightly packed cells. Structural thermal gradients (due to localized cooling flow resistance or edge heat rejection) induce uneven current density distribution. This spatial variance accelerates localized over-stress, driving individual cell degradation and cascade pack failure.

### 3. Cell Voltage Spread

$$\Delta V_{\text{cell}} = V_{\text{cell, max}} - V_{\text{cell, min}} \quad \left(\text{or } \frac{\text{voltage\_imbalance}}{1000} \text{ V}\right)$$

- **Physical Justification:** Open-circuit and loaded cell voltage divergences signify unequal internal impedance and loss of usable capacity. As a pack deteriorates, weaker cells reach voltage cutoff limits prematurely during discharge, triggering BMS over-discharge protection or thermal instability.

### 4. Fast-Charge Cumulative Stress Index

$$\text{Stress}_{\text{fast-charge}} = \text{Fast Charge Ratio} \times \text{Cycle Count}$$

- **Physical Justification:** High C-rate DC fast charging exposes graphite anode particles to severe concentration polarization, mechanical strain, and particle micro-cracking. Weighting total cycle count by fast-charging fraction isolates cumulative mechanical fatigue on electrode active materials.

### 5. Resistance Growth Proxy

$$R_{\text{growth}} = R_{\text{internal}} \times \left(1 + \frac{\text{Capacity Loss \%}}{100}\right)$$

- **Physical Justification:** Both capacity fade (loss of recyclable lithium) and resistance rise (ohmic and charge-transfer impedance from SEI thickening) co-evolve during cell aging. This interaction proxy amplifies the Joule heating ($P_{\text{loss}} = I^2 R$) vulnerability of aged packs under heavy power draws.

### 6. Equivalent Full Cycles (EFC) per 1,000 km

$$\text{EFC}_{1000\text{km}} = \left( \frac{\text{Cycle Count}}{\max(\text{Odometer}_{\text{km}}, 1.0)} \right) \times 1000$$

- **Physical Justification:** Normalizes battery cycling against vehicular travel. Vehicles operating with high EFC per km represent short-range, aggressive start-stop, or high auxiliary-load operating regimes, experiencing higher cyclical wear per unit distance.

### 7. Composite Physical Stress Aggregate

$$\text{Stress}_{\text{composite}} = \text{Stress}_{\text{thermal}} + \text{Stress}_{\text{charging}} + \text{Stress}_{\text{driving}}$$

- **Physical Justification:** Battery degradation is rarely uni-dimensional. Degradation accelerates when thermal abuse, high charging stress, and aggressive driver acceleration occur simultaneously. This composite metric provides a holistic degradation score.

### 8. State-of-Health (SoH) Wear Loss per Cycle

$$\Delta \text{SoH}_{\text{cycle}} = \frac{100 - \text{Battery Health \%}}{\max(\text{Cycle Count}, 1)}$$

- **Physical Justification:** Measures the differential wear slope ($\partial \text{SoH} / \partial N$). Batteries with elevated wear per cycle indicate premature cell degradation or manufacturing batch defects long before absolute health drops below standard replacement thresholds.

---

### 🌟 Novel Electrochemical Degradation Features

- **9. Thermal Dissipation Strain:**
  $$\text{Strain}_{\text{cooling}} = \frac{T_{\text{cell, max}} - T_{\text{ambient}}}{\max(\text{Cooling Health}, 1.0)}$$
  Measures cooling system insufficiency under peak thermal generation.
- **10. Internal Resistance per kWh Capacity:**
  $$\text{IR}_{\text{density}} = \frac{R_{\text{internal}}}{\max(\text{Capacity}_{\text{kWh}}, 1.0)}$$
  Standardizes cell impedance across diverse battery capacities (e.g. 30 kWh vs 130 kWh).
- **11. Lithium Plating Susceptibility Metric:**
  $$\text{Risk}_{\text{plating}} = \max(0, 15 - T_{\text{ambient}}) \times \text{Fast Charge Ratio} \times \left(\frac{\text{SoC}}{100}\right)$$
  Detects the high-risk thermodynamic regime where rapid lithium insertion into graphite at sub-15°C temperatures causes metallic lithium dendrite deposition.
- **12. Degradation Acceleration Ratio:**
  $$\text{Deg}_{\text{accel}} = \frac{\text{Capacity Loss \%}}{\max(\text{Vehicle Age}_{\text{years}}, 0.1)}$$
  Captures annual calendar aging velocity.

---

## 📊 Model Performance & Evaluation

The dataset exhibits an imbalanced target distribution (~6.92% failure rate). The XGBoost classifier was configured with `scale_pos_weight = 13.45` to prioritize minority class failure recall while sustaining high precision:

| Metric | Test Set Score | Description |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.9797** | True vs False positive discrimination across all thresholds |
| **PR-AUC (Avg Precision)** | **0.7585** | ~11x lift over random baseline (0.0692) |
| **Failure Recall** | **85.2%** | Successfully flags 85 out of every 100 failing packs |
| **Overall Accuracy** | **95.3%** | Accurate classification across 4,000 holdout packs |

### Confusion Matrix (Test Set: 4,000 Packs)

- **True Negatives (Healthy):** 3,578
- **False Positives:** 145
- **False Negatives:** 41
- **True Positives (Failure Detected):** 236

---

## 💻 Streamlit UI Dashboard (`app.py`)

The application provides a modern dark-mode diagnostic interface:

1. **Tab 1: Real-Time Diagnostic Dashboard**
   - Live interactive Gauge Chart displaying predicted failure probability (0–100%).
   - Color-coded risk status badges:
     - 🟢 **Low Risk (<20%):** Nominal operating envelope.
     - 🟡 **Moderate Risk (20%–55%):** Early degradation flags, proactive balancing recommended.
     - 🔴 **High Risk (>55%):** Critical imminent failure alert with actionable field mitigation.
   - Real-time physics KPI metric cards and multi-dimensional stress radar chart.

2. **Tab 2: Feature Importance & Analytics**
   - Global XGBoost gain importance bar chart with color differentiation for physics-engineered metrics.
   - Validation performance summary cards and electrochemical sensitivity insights.

3. **Tab 3: Fleet Batch Processing Engine**
   - Batch upload of raw fleet telemetry CSV files (or one-click 1,000-vehicle benchmark load).
   - High-throughput calculation of all 12 physics features across the fleet.
   - Fleet risk breakdown KPIs, risk donut chart, and probability distribution histogram.
   - Filterable telemetry table with one-click export for high-risk vehicles (`high_risk_ev_battery_telemetry.csv`).

---

## 🚀 Getting Started

### 1. Requirements

```bash
pip install pandas numpy scikit-learn xgboost streamlit plotly matplotlib seaborn joblib
```

### 2. Train Model & Export Pipeline

```bash
python eda_and_model.py
```

This generates:

- `ev_battery_model.pkl` (Trained XGBoost Classifier)
- `scaler.pkl` (StandardScaler)
- `imputer.pkl` (Median Imputer)
- `feature_columns.pkl` (Feature column names list)
- `ev_pipeline_bundle.pkl` (Self-contained inference bundle)
- `model_evaluation_metrics.png` (ROC, PR, and Confusion Matrix charts)
- `feature_importance.png` (Top 20 features importance chart)

### 3. Launch Streamlit UI

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.
