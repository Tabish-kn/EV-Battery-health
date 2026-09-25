# ⚡ EV Battery Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-EB5424.svg?style=flat)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-blueviolet.svg?style=flat)](https://shap.readthedocs.io/)
[![Pytest](https://img.shields.io/badge/pytest-Passing-brightgreen.svg?style=flat&logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

An end-to-end, production-grade **Electric Vehicle (EV) Battery Health Prognostics, Predictive Maintenance, SOH/RUL Forecasting, Unsupervised Anomaly Detection, and Explainable AI (XAI)** platform.

---

## 📑 Table of Contents

1. [Executive Overview](#1-executive-overview)
2. [Key Platform Features](#2-key-platform-features)
3. [System Architecture](#3-system-architecture)
4. [Dataset & Physical Telemetry](#4-dataset--physical-telemetry)
5. [Domain-Physics Feature Engineering](#5-domain-physics-feature-engineering)
6. [Machine Learning Models & Benchmark Performance](#6-machine-learning-models--benchmark-performance)
7. [State of Health (SOH) Estimation](#7-state-of-health-soh-estimation)
8. [Remaining Useful Life (RUL) Prediction](#8-remaining-useful-life-rul-prediction)
9. [Unsupervised Anomaly Detection](#9-unsupervised-anomaly-detection)
10. [Multi-Factor Composite Risk Index](#10-multi-factor-composite-risk-index)
11. [Explainable AI (XAI) & SHAP Interpretability](#11-explainable-ai-xai--shap-interpretability)
12. [What-If Counterfactual Simulator](#12-what-if-counterfactual-simulator)
13. [Interactive Streamlit Dashboard](#13-interactive-streamlit-dashboard)
14. [Batch Telemetry Processing & Diagnostic Reports](#14-batch-telemetry-processing--diagnostic-reports)
15. [Installation & Setup](#15-installation--setup)
16. [Running the Platform](#16-running-the-platform)
17. [Jupyter Notebooks Experimentation Suite](#17-jupyter-notebooks-experimentation-suite)
18. [Testing & Quality Assurance](#18-testing--quality-assurance)
19. [Docker Deployment](#19-docker-deployment)
20. [Project Directory Layout](#20-project-directory-layout)
21. [Scientific Limitations & Ethical Governance](#21-scientific-limitations--ethical-governance)
22. [Roadmap & Future Work](#22-roadmap--future-work)
23. [License & Citation](#23-license--citation)

---

## 1. Executive Overview

The **EV Battery Intelligence Platform** transforms high-dimensional electric vehicle battery management system (BMS) telemetry into actionable prognostic and diagnostic insights. By coupling **domain electrochemical kinetics** (e.g. Arrhenius aging, thermal dissipation strain, lithium plating susceptibility) with **state-of-the-art machine learning algorithms**, the platform provides:

- **Early Failure Warning:** Calibrated classification targeting battery failure precursors with **92.06% safety recall** (ROC-AUC: 0.9826, PR-AUC: 0.7809).
- **Continuous Capacity Tracking:** State of Health (SOH) estimation achieving **$R^2 = 0.9254$** and **$\text{MAE} = 1.95\%$ SOH**.
- **Longitudinal Cycle Life Prognostics:** Remaining Useful Life (RUL) forecasting with empirical 90% confidence prediction intervals (**$R^2 = 0.8964$**, $\text{MAE} = 425$ cycles).
- **Label-Free Anomaly Isolation:** Multivariate Isolation Forest scoring pack drifts, sudden impedance spikes, and voltage divergences.
- **Explainable Decision Support:** TreeSHAP attributions isolating the positive risk drivers and negative mitigators for every individual battery pack.

---

## 2. Key Platform Features

| Module | Core Capability | Engineering Significance |
| :--- | :--- | :--- |
| **⚡ Multi-Target Model Suite** | Unified inference for Failure, SOH %, RUL cycles, and Anomaly severity. | Comprehensive single-pane diagnostics for fleet operators and engineers. |
| **🔋 Electrochemical Physics Engine** | 17+ domain-physics metrics derived from raw telemetry without target leakage. | Captures non-linear thermal acceleration, SEI growth, and mechanical stress. |
| **🛡️ Automated Data Quality Pipeline** | Ingestion validation, physical boundary enforcement, and spike clipping. | Prevents silent data corruption and invalid sensor spikes from skewing models. |
| **🔍 Explainable AI (XAI)** | TreeSHAP local waterfall/force charts and global feature importance. | Demystifies black-box models; provides transparent reasoning for maintenance. |
| **🧪 What-If Scenario Simulator** | Counterfactual parameter modification with real-time delta tracking. | Enables fleet managers to simulate high-temperature routes or fast-charging loads. |
| **📊 Fleet Telemetry Analytics** | Multi-attribute filtering across Brand, Chemistry (NMC, LFP, NCA, LTO, LMO), and Drive Type. | Uncovers cohort-level degradation patterns across operating environments. |
| **📄 Automated Diagnostic Reports** | Instant generation of certified, printable HTML diagnostic certificates & CSV exports. | Ready for field service technicians, warranty claims, and maintenance records. |

---

## 3. System Architecture

```text
+-----------------------------------------------------------------------------------+
|                           RAW BATTERY TELEMETRY (CSV / BMS)                       |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                        1. DATA VALIDATION & SPIKE CLIPPING                        |
|   - Physical Boundary Checks   - Spike Clipping   - Target Leakage Auditing       |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                     2. ELECTROCHEMICAL & PHYSICS FEATURE ENGINE                   |
|   - Arrhenius Factor    - Voltage Spread      - Fast-Charge Stress Index          |
|   - Resistance Proxy    - Plating Risk        - Thermal Dissipation Strain        |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                     3. STANDARDIZED PREPROCESSING PIPELINE                        |
|   - Median Numeric Imputation   - OneHotEncoding   - StandardScaler               |
+-----------------------------------------------------------------------------------+
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
+-------------------+           +-------------------+           +-------------------+
| FAILURE CLASSIFIER|           |   SOH REGRESSOR   |           |   RUL REGRESSOR   |
| (PR-AUC: 0.7809)  |           |   (R²: 0.9254)    |           |   (R²: 0.8964)    |
| Recall: 92.06%    |           |   MAE: 1.95% SOH  |           |   MAE: 425 cycles |
+-------------------+           +-------------------+           +-------------------+
         │                               │                               │
         └───────────────────────────────┼───────────────────────────────┘
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                       4. UNSUPERVISED ANOMALY DETECTOR                            |
|             Isolation Forest (Contamination=5%) -> 0-100 Anomaly Severity         |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                     5. COMPOSITE 0-100 BATTERY RISK SCORING                       |
|   Risk Score = 0.30*Fail + 0.20*Thermal + 0.15*Elect + 0.15*Chg + 0.10*Deg + 0.10*Anom  |
+-----------------------------------------------------------------------------------+
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
+-------------------+           +-------------------+           +-------------------+
|  EXPLAINABLE AI   |           | WHAT-IF SIMULATOR |           | STREAMLIT UI &    |
|  (TreeSHAP XAI)   |           | (Counterfactuals) |           | DIAGNOSTIC REPORT |
+-------------------+           +-------------------+           +-------------------+
```

---

## 4. Dataset & Physical Telemetry

The platform is evaluated on a benchmark telemetry dataset of **20,000 electric vehicle battery records** spanning **70 physical and operational parameters**:

- **Dimensionality:** 20,000 records × 70 columns (60 continuous numerical sensors, 10 categorical descriptors).
- **Target Distribution (Failure):** 1,384 failures (6.92%), 18,616 healthy packs (93.08%) — 13.45:1 imbalance ratio.
- **SOH Distribution:** Mean 81.88% ± 9.11% (Range: 47.77% – 100.00%).
- **RUL Distribution:** Mean 7,825 cycles ± 1,680 cycles (Range: 1,656 – 12,870 cycles).
- **Physical Boundary Enforcement:**
  - Pack Voltage: $100.0\,\text{V} - 900.0\,\text{V}$
  - Cell Voltage: $2.0\,\text{V} - 4.5\,\text{V}$
  - Cell Temperature: $-30.0^\circ\text{C} - 85.0^\circ\text{C}$
  - Ambient Temperature: $-40.0^\circ\text{C} - 60.0^\circ\text{C}$
  - Internal Resistance: $0.01\,\text{m}\Omega - 5.0\,\text{m}\Omega$

---

## 5. Domain-Physics Feature Engineering

The feature engine ([`src/feature_engine.py`](file:///a:/Sem%202/files/EV/src/feature_engine.py)) computes 17 domain-physics indicators that augment machine learning estimators while strictly preventing target leakage:

1. **Arrhenius Thermal Aging Acceleration Factor:**
   $$k_{\text{Arrhenius}} = \exp\left(\frac{\max(0, \max(T_{\text{ambient}}, T_{\text{cell,max}}) - 25)}{10.0}\right)$$
2. **Cell Temperature Variance Delta:**
   $$\Delta T_{\text{cell}} = T_{\text{cell,max}} - T_{\text{cell,min}}$$
3. **Cell Voltage Spread & Imbalance:**
   $$\Delta V_{\text{spread}} = V_{\text{cell,max}} - V_{\text{cell,min}} \quad (\text{or } \Delta V_{\text{imbalance}} / 1000)$$
4. **Fast-Charge Cumulative Stress Index:**
   $$\text{Stress}_{\text{fast\_charge}} = \text{FastChargeRatio} \times \text{CycleCount}$$
5. **Resistance Growth Proxy:**
   $$R_{\text{growth}} = R_{\text{internal}} \times \left(1 + \frac{\text{CapacityLoss}_{\%}}{100}\right)$$
6. **Equivalent Full Cycles (EFC) per 1,000 km:**
   $$\text{EFC}_{1000\text{km}} = \left(\frac{\text{CycleCount}}{\max(\text{Odometer}_{\text{km}}, 1)}\right) \times 1000$$
7. **Composite Physical Stress Aggregate:**
   $$\text{Stress}_{\text{composite}} = \text{Stress}_{\text{thermal}} + \text{Stress}_{\text{charging}} + \text{Stress}_{\text{driving}}$$
8. **Thermal Dissipation Strain:**
   $$\text{Strain}_{\text{thermal}} = \frac{T_{\text{cell,max}} - T_{\text{ambient}}}{\max(\text{CoolingHealth}, 1.0)}$$
9. **Internal Resistance per kWh Capacity:**
   $$R_{\text{density}} = \frac{R_{\text{internal}}}{\max(\text{Capacity}_{\text{kWh}}, 1.0)}$$
10. **Lithium Plating Susceptibility Metric:**
    $$\text{Risk}_{\text{plating}} = \max(0, 15.0 - T_{\text{ambient}}) \times \text{FastChargeRatio} \times \left(\frac{\text{SOC}}{100}\right)$$
11. **Degradation Acceleration Index:**
    $$\text{DAI} = \frac{\text{CapacityLoss}_{\%}}{\max(\text{VehicleAge}_{\text{years}}, 0.1)}$$
12. **C-Rate Proxy Indicator:**
    $$\text{C-Rate}_{\text{proxy}} = \frac{\text{ChargePower}_{\text{kW}}}{\max(\text{Capacity}_{\text{kWh}}, 1.0)}$$
13. **Depth of Discharge (DOD) Fast-Charge Stress:**
    $$\text{DOD}_{\text{stress}} = \text{DOD} \times (1.0 + \text{FastChargeRatio})$$
14. **Electrical Efficiency Loss:**
    $$\text{Loss}_{\text{efficiency}} = (100.0 - \eta_{\text{charge}}) + (100.0 - \eta_{\text{discharge}})$$
15. **Thermal Severity Index:**
    $$\text{TSI} = T_{\text{cell,max}} \times \left(1.0 + \frac{\text{ThermalRunawayRisk}}{100.0}\right)$$
16. **BMS Fault Density:**
    $$\text{FaultDensity} = \frac{\text{BMSWarnings} + \text{SensorFaults} + \text{AbnormalVoltageEvents}}{\max(\text{VehicleAge}_{\text{years}}, 0.5)}$$

---

## 6. Machine Learning Models & Benchmark Performance

All models were evaluated on a dedicated **20% stratified holdout test split** ($n = 4,000$ samples) using Scikit-Learn and XGBoost:

### 6.1 Failure Classification Benchmark

| Classifier | ROC-AUC | PR-AUC | F1-Score | Recall | Precision | Accuracy | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Calibrated Logistic** | **0.9826** | **0.7809** | **0.6977** | **92.06%** | **56.17%** | **94.48%** | **0.0408** |
| **Gradient Boosting** | 0.9800 | 0.7646 | 0.6554 | 55.96% | 79.08% | 95.93% | 0.0282 |
| **XGBoost Classifier** | 0.9792 | 0.7586 | 0.7178 | 84.48% | 62.40% | 95.40% | 0.0322 |
| **Random Forest** | 0.9555 | 0.6232 | 0.5658 | 56.68% | 56.47% | 93.97% | 0.0417 |

> **Evaluation Decision:** The Calibrated Logistic / XGBoost pipeline was selected as champion due to its superior **PR-AUC (0.7809)** and **92.06% safety recall**, ensuring critical battery failure events are rarely missed.

---

## 7. State of Health (SOH) Estimation

The SOH regression engine predicts capacity retention percentage without direct target leakage:

| SOH Regressor | $R^2$ Score | MAE (% SOH) | RMSE (% SOH) | MAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost Regressor** | **0.9254** | **1.95%** | **2.46%** | **2.42%** |
| **Gradient Boosting** | 0.9184 | 2.05% | 2.58% | 2.54% |
| **Random Forest** | 0.9141 | 2.09% | 2.64% | 2.60% |
| **Ridge Regression** | 0.9095 | 2.15% | 2.71% | 2.67% |

---

## 8. Remaining Useful Life (RUL) Prediction

The RUL regression engine projects remaining full equivalent cycles until the pack degrades to the 70% SOH End-of-Life (EOL) threshold. It integrates empirical 90% confidence intervals:

| RUL Regressor | $R^2$ Score | MAE (Cycles) | RMSE (Cycles) | MAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost Regressor** | **0.8964** | **425.09** | **530.56** | **5.78%** |
| **Gradient Boosting** | 0.8958 | 426.02 | 532.04 | 5.78% |
| **Ridge Regression** | 0.8953 | 426.32 | 533.47 | 5.81% |
| **Random Forest** | 0.8946 | 429.90 | 535.26 | 5.84% |

---

## 9. Unsupervised Anomaly Detection

To detect novel sensor failures, thermal spikes, and cell imbalances without labeled failure instances, the platform uses an **Isolation Forest (150 trees, 5% contamination rate)**:

- **Contamination Rate:** 5.00% ($1,000 / 20,000$ packs isolated).
- **Normalized Scale:** $[0.0, 100.0]$ where $0 = \text{Nominal}$ and $100 = \text{Severe Outlier}$.
- **Attribution Engine:** Calculates multivariate $Z$-score deviations against fleet nominal baselines to rank the sensors contributing most to the anomaly.

---

## 10. Multi-Factor Composite Risk Index

The composite battery risk score ($0 - 100$) aggregates six weighted diagnostic sub-indices:

$$\text{Risk Score} = 0.30 \cdot R_{\text{failure}} + 0.20 \cdot R_{\text{thermal}} + 0.15 \cdot R_{\text{electrical}} + 0.15 \cdot R_{\text{charging}} + 0.10 \cdot R_{\text{degradation}} + 0.10 \cdot R_{\text{anomaly}}$$

### Diagnostic Risk Tiers

- 🟢 **LOW Risk (0 – 29.9):** Nominal operation; standard maintenance cycle.
- 🟡 **MODERATE Risk (30 – 59.9):** Elevated thermal or fast-charging stress; diagnostic review recommended.
- 🟠 **HIGH Risk (60 – 79.9):** Significant impedance growth or cell imbalance; service inspection required.
- 🔴 **CRITICAL Risk (80 – 100):** Imminent thermal runaway or severe capacity loss; immediate pack isolation.

---

## 11. Explainable AI (XAI) & SHAP Interpretability

The platform integrates **TreeSHAP** to provide both global model transparency and localized explanations for individual pack diagnostic decisions:

- **Global Attribution:** Ranks overall telemetry importance across the fleet (identifying thermal runaway risk, cell temperature max, internal resistance, and capacity loss as primary risk drivers).
- **Local Waterfall Attributions:** Quantifies exact positive SHAP contributions (factors driving the pack toward failure risk) and negative SHAP contributions (factors preserving pack health).
- **Governance Notice:** SHAP values describe statistical model attribution, not electrochemical causality.

---

## 12. What-If Counterfactual Simulator

The simulator ([`src/what_if.py`](file:///a:/Sem%202/files/EV/src/what_if.py)) allows engineers to adjust operating parameters interactively and observe the downstream impact across all four intelligence models:

```text
BASELINE STATE                        SIMULATED SCENARIO
─────────────────                     ──────────────────
Ambient Temp: 25°C       ───►         Ambient Temp: 45°C
Fast Charge:  30%        ───►         Fast Charge:  80%
Cooling:      80%        ───►         Cooling:      40%

RESULTS & DELTAS:
- Failure Probability:   3.2%   ──►   18.6%   (+15.4% pts)
- SOH Retention:         84.2%  ──►   82.8%   (-1.4% pts)
- RUL Cycles:            6,200  ──►   4,850   (-1,350 cycles)
- Composite Risk Score:  24.1   ──►   62.8    (+38.7 pts, HIGH RISK)
```

---

## 13. Interactive Streamlit Dashboard

The production Streamlit UI ([`app.py`](file:///a:/Sem%202/files/EV/app.py)) provides 10 dedicated diagnostic views:

1. ⚡ **Executive Overview:** High-level KPI cards, risk gauge, radar chart, and pack subsystem health status.
2. 🔋 **Battery Health & SOH:** Continuous capacity retention, Arrhenius aging curves, and degradation trajectories.
3. ⚠️ **Failure Prediction:** Model failure probability gauge, classification threshold tuning, and risk factors.
4. ⏳ **Remaining Useful Life (RUL):** Cycle forecasting with 90% confidence prediction intervals.
5. 🚨 **Anomaly Detection:** Isolation Forest score (0-100), anomaly flag, and sensor $Z$-score radar attributions.
6. 🔍 **Explainable AI (SHAP):** Global summary plots, local positive/negative feature contribution charts.
7. 🧪 **What-If Simulator:** Real-time parameter sliders with baseline vs. scenario comparison cards.
8. 📊 **Fleet Analytics:** Multi-filter engine across vehicle brand, battery chemistry, and drive type.
9. 📄 **Batch Processing & Reports:** CSV batch upload, live validation, enriched CSV export, and HTML diagnostic certificate generation.
10. 📈 **Model Performance & Cards:** Consolidated evaluation tables, confusion matrices, ROC/PR curves, and markdown Model Card viewer.

---

## 14. Batch Telemetry Processing & Diagnostic Reports

Users can upload raw vehicle telemetry CSV files for automated processing:

1. **Upload & Validation:** Schema integrity, duplicate checking, and boundary compliance.
2. **Inference Execution:** Simultaneous scoring across Failure, SOH, RUL, Anomaly, and Risk engines.
3. **Data Export:** Download the enriched batch CSV containing all diagnostic predictions.
4. **HTML Diagnostic Certificate:** Generates a certified, self-contained HTML diagnostic audit report with styling and summary tables for individual battery packs.

---

## 15. Installation & Setup

### 15.1 Prerequisites

- Python 3.10, 3.11, or 3.12
- Git

### 15.2 Clone and Environment Creation

```bash
# 1. Clone repository
git clone https://github.com/Tabish-kn/EV-Battery-health.git
cd EV-Battery-health

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows (PowerShell / CMD):
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 16. Running the Platform

### 16.1 Train All Models & Serialize Artifacts

To run the full model training pipeline (Failure classifier, SOH regressor, RUL regressor, Anomaly detector, SHAP summaries):

```bash
python train_all_models.py
```

### 16.2 Launch the Interactive Streamlit App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## 17. Jupyter Notebooks Experimentation Suite

The `notebooks/` directory contains 10 structured, reproducible Jupyter notebooks:

- [`01_data_audit.ipynb`](file:///a:/Sem%202/files/EV/notebooks/01_data_audit.ipynb): Automated dataset quality, missingness, and boundary audit.
- [`02_eda.ipynb`](file:///a:/Sem%202/files/EV/notebooks/02_eda.ipynb): Exploratory telemetry, correlation heatmaps, and distributions.
- [`03_feature_engineering.ipynb`](file:///a:/Sem%202/files/EV/notebooks/03_feature_engineering.ipynb): Domain-physics electrochemical derivations.
- [`04_failure_prediction.ipynb`](file:///a:/Sem%202/files/EV/notebooks/04_failure_prediction.ipynb): Classifier training, benchmark comparison, and ROC/PR curves.
- [`05_soh_prediction.ipynb`](file:///a:/Sem%202/files/EV/notebooks/05_soh_prediction.ipynb): SOH continuous regression and residual diagnostics.
- [`06_rul_prediction.ipynb`](file:///a:/Sem%202/files/EV/notebooks/06_rul_prediction.ipynb): RUL cycle life modeling with prediction intervals.
- [`07_anomaly_detection.ipynb`](file:///a:/Sem%202/files/EV/notebooks/07_anomaly_detection.ipynb): Isolation Forest multivariate anomaly isolation.
- [`08_degradation_forecasting.ipynb`](file:///a:/Sem%202/files/EV/notebooks/08_degradation_forecasting.ipynb): Arrhenius capacity fade projections.
- [`09_explainable_ai.ipynb`](file:///a:/Sem%202/files/EV/notebooks/09_explainable_ai.ipynb): TreeSHAP global and local waterfall explanations.
- [`10_model_comparison.ipynb`](file:///a:/Sem%202/files/EV/notebooks/10_model_comparison.ipynb): Multi-model benchmark consolidation.

---

## 18. Testing & Quality Assurance

Run the automated test suite using `pytest`:

```bash
python -m pytest tests/ -v
```

### Test Coverage

- [`tests/test_preprocessing.py`](file:///a:/Sem%202/files/EV/tests/test_preprocessing.py): Preprocessor fitting, median imputation, one-hot encoding, and scaling.
- [`tests/test_features.py`](file:///a:/Sem%202/files/EV/tests/test_features.py): Physics formulas, Arrhenius math, sensor spike clipping, and leakage prevention.
- [`tests/test_models.py`](file:///a:/Sem%202/files/EV/tests/test_models.py): ModelBundle loading, risk scoring formulas, and anomaly detection logic.
- [`tests/test_predictions.py`](file:///a:/Sem%202/files/EV/tests/test_predictions.py): End-to-end single and batch inference pipelines.

---

## 19. Docker Deployment

### 19.1 Build Docker Image

```bash
docker build -t ev-battery-intelligence:2.0.0 .
```

### 19.2 Run Docker Container

```bash
docker run -p 8501:8501 ev-battery-intelligence:2.0.0
```

Access the containerized dashboard at `http://localhost:8501`.

---

## 20. Project Directory Layout

```text
EV-Battery-health/
├── data/
│   ├── raw/                  <- Raw ev_battery_health.csv dataset (20,000 rows)
│   ├── processed/            <- Cleaned, feature-engineered datasets
│   └── sample/               <- Curated 100-row sample telemetry for demo/testing
├── notebooks/
│   ├── 01_data_audit.ipynb   <- Data quality and missingness audit
│   ├── 02_eda.ipynb          <- Exploratory data analysis
│   ├── 03_feature_engineering.ipynb <- Physics derivations
│   ├── 04_failure_prediction.ipynb  <- Failure classifier benchmark
│   ├── 05_soh_prediction.ipynb      <- SOH regression
│   ├── 06_rul_prediction.ipynb      <- RUL prediction & intervals
│   ├── 07_anomaly_detection.ipynb   <- Isolation Forest anomaly engine
│   ├── 08_degradation_forecasting.ipynb <- Degradation curves
│   ├── 09_explainable_ai.ipynb      <- TreeSHAP explanations
│   └── 10_model_comparison.ipynb   <- Model benchmarking
├── src/
│   ├── __init__.py           <- Package initializer (v2.0.0)
│   ├── config.py             <- Paths, physical boundaries, and hyperparameters
│   ├── data_loader.py        <- Data ingestion and train/test splitting
│   ├── validation.py         <- Quality checks, spike clipping, target leakage tests
│   ├── preprocessing.py      <- Transformer pipelines (BatteryPreprocessor)
│   ├── feature_engine.py     <- 17+ domain-physics feature derivations
│   ├── models.py             <- Unified ModelBundle serialization container
│   ├── failure_prediction.py <- Failure classification pipeline
│   ├── soh_prediction.py     <- SOH regression pipeline
│   ├── rul_prediction.py     <- RUL regression pipeline & intervals
│   ├── anomaly_detection.py  <- Isolation Forest anomaly scoring
│   ├── degradation.py        <- Arrhenius kinetics & degradation curves
│   ├── risk_scoring.py       <- Composite 6-factor risk index
│   ├── explainability.py     <- SHAP explainer for global & local XAI
│   ├── what_if.py            <- Counterfactual scenario simulator
│   └── utils.py              <- Batch inference runner & HTML report generator
├── models/
│   ├── failure_model/        <- Serialized classifier & metadata JSON
│   ├── soh_model/            <- Serialized SOH regressor & metadata JSON
│   ├── rul_model/            <- Serialized RUL regressor & metadata JSON
│   ├── anomaly_model/        <- Serialized Isolation Forest & metadata JSON
│   └── preprocessors/        <- Scalers, imputers, and encoder artifacts
├── reports/
│   ├── figures/              <- Generated high-res evaluation plots
│   └── generated/            <- Exported battery health audit reports
├── tests/
│   ├── conftest.py           <- Pytest configuration
│   ├── test_preprocessing.py <- Preprocessing & imputer unit tests
│   ├── test_features.py      <- Physics math & spike clipping tests
│   ├── test_models.py        <- Model loading & risk score tests
│   └── test_predictions.py   <- End-to-end inference & batch pipeline tests
├── app.py                    <- Production Streamlit intelligence dashboard
├── train_all_models.py       <- Master training pipeline for all models
├── requirements.txt          <- Pinned dependency specification
├── MODEL_CARD.md             <- Comprehensive Model Card documentation
├── PROJECT_AUDIT.md          <- Codebase and dataset audit document
├── Dockerfile                <- Containerized deployment configuration
└── README.md                 <- System documentation & operational handbook
```

---

## 21. Scientific Limitations & Ethical Governance

1. **Statistical Attribution vs. Physical Causation:** Machine learning predictions and SHAP values quantify statistical correlations present in historical telemetry. They represent model behavior and do not constitute physical or electrochemical causal proof.
2. **Pack-Level Telemetry Resolution:** Telemetry features represent pack and module-level sensors. Sub-millisecond cell impedance spectroscopy or microscopic physical cell teardowns are not present in this dataset.
3. **Engineering Decision Support:** This platform is designed as an advisory decision-support tool. It does not replace primary hardware BMS safety cutoffs, pyrofuse disconnects, or physical technician inspections.

---

## 22. Roadmap & Future Work

- [ ] **Cell-Level Digital Twin:** Expansion to individual cell-level series/parallel topology modeling when sub-cell telemetry is available.
- [ ] **Edge BMS Deployment:** Quantization and ONNX export for low-power microcontroller deployment (ARM Cortex-M / STM32).
- [ ] **Fleet Telematics Streaming:** WebSocket integration for real-time MQTT/Kafka streaming telemetry ingestion.
- [ ] **Physics-Informed Neural Networks (PINNs):** Integration of differential electrochemical equations into neural network loss functions.

---

## 23. License & Citation

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```bibtex
@software{ev_battery_intelligence_2026,
  author = {Tabish-kn},
  title = {EV Battery Intelligence Platform: Prognostics, Predictive Maintenance, SOH/RUL Forecasting, and Explainable AI},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Tabish-kn/EV-Battery-health}
}
```
