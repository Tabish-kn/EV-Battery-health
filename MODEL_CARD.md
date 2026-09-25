# Machine Learning Model Card: EV Battery Intelligence Platform

**Document Version:** 2.0.0  
**Release Date:** 2026-09-25  
**Platform:** EV Battery Health, SOH/RUL, Predictive Maintenance & Anomaly Detection Suite  
**Maintainer:** AI/ML Diagnostics & Powertrain Engineering Group

---

## 1. Overview of Platform Models

The EV Battery Intelligence Platform integrates four specialized machine learning models and one deterministic composite risk scoring engine:

| Model ID | Model Class | Primary Algorithm | Target / Output | Evaluation Metric |
| :--- | :--- | :--- | :--- | :--- |
| **`failure_classifier`** | Binary Classification | Gradient Boosted / Calibrated Logistic | Probability of pack failure (`0` to `1`) | PR-AUC = 0.7809, ROC-AUC = 0.9826 |
| **`soh_regressor`** | Continuous Regression | XGBoost Regressor | State of Health percentage (`SOH %`) | R² = 0.9254, MAE = 1.95% SOH |
| **`rul_regressor`** | Continuous Regression | XGBoost Regressor + Prediction Intervals | Remaining Useful Cycles (`RUL`) | R² = 0.8964, MAE = 425 cycles |
| **`anomaly_detector`** | Unsupervised Detection | Isolation Forest (5% Contamination) | Anomaly Score (`0-100`) & Flag | 5.00% Anomaly Rate (1,000/20,000) |
| **`battery_risk_index`** | Multi-Factor Scoring | Physics-Weighted Composite Index | Risk Score (`0-100`) & Category | Calibrated [0, 100] Boundary |

---

## 2. Model 1: Battery Failure Prediction Classifier (`failure_classifier`)

### 2.1 Model Details & Architecture

- **Purpose:** Early detection of critical battery pack failure, thermal runaway risk, and severe cell degradation.
- **Algorithm:** Calibrated Classifier / XGBoost with cost-sensitive class balancing (`scale_pos_weight` = 13.45).
- **Inputs:** Processed features (electrical, thermal, mechanical telemetry + one-hot encoded metadata + 17 physics-engineered indicators).
- **Output:** Calibrated failure probability ($P \in [0.0, 1.0]$) and binary classification flag at $0.50$ threshold.

### 2.2 Training & Validation Data

- **Dataset:** 20,000 vehicle telemetry diagnostic records (`ev_battery_health.csv`).
- **Split Strategy:** 80% Train ($n=16,000$), 20% Holdout Test ($n=4,000$), stratified by target class.
- **Class Balance:** 1,384 failures (6.92%), 18,616 healthy (93.08%).

### 2.3 Performance Metrics (Holdout Test Set)

- **ROC-AUC:** 0.9826
- **PR-AUC (Precision-Recall AUC):** 0.7809 (vs. baseline prevalence 6.92%)
- **F1-Score:** 0.7178 (XGBoost) / 0.6977 (Calibrated Classifier)
- **Recall (Sensitivity):** 92.06%
- **Precision:** 56.17% (tuned for high safety recall)
- **Brier Score Loss:** 0.0322 – 0.0408

---

## 3. Model 2: State of Health (SOH) Regressor (`soh_regressor`)

### 3.1 Model Details & Architecture

- **Purpose:** Estimates battery capacity retention as a percentage of nominal rated capacity ($\text{SOH} = \frac{C_{\text{current}}}{C_{\text{nominal}}} \times 100\%$).
- **Algorithm:** XGBoost Regressor (300 estimators, learning rate = 0.05, max depth = 5).
- **Leakage Prevention:** Direct capacity loss percentages, SOH duplicate labels, and failure targets are strictly excluded from the feature matrix during training.
- **Output:** Estimated continuous State of Health percentage ($[0.0\%, 100.0\%]$).

### 3.2 Performance Metrics (Holdout Test Set)

- **R² Score:** 0.9254
- **Mean Absolute Error (MAE):** 1.95% SOH
- **Root Mean Squared Error (RMSE):** 2.46% SOH
- **Mean Absolute Percentage Error (MAPE):** 2.42%

---

## 4. Model 3: Remaining Useful Life (RUL) Regressor (`rul_regressor`)

### 4.1 Model Details & Architecture

- **Purpose:** Estimates remaining full equivalent charge/discharge cycles before battery capacity reaches the End-of-Life (EOL) threshold (70–80% SOH retention).
- **Algorithm:** XGBoost Regressor wrapped with empirical residual-calibrated prediction intervals ($90\%$ confidence bounds).
- **Output:** Point prediction of remaining cycles + lower and upper confidence bounds.

### 4.2 Performance Metrics (Holdout Test Set)

- **R² Score:** 0.8964
- **MAE:** 425.09 cycles
- **RMSE:** 530.56 cycles
- **MAPE:** 5.78%

---

## 5. Model 4: Unsupervised Anomaly Detector (`anomaly_detector`)

### 5.1 Model Details & Architecture

- **Purpose:** Identifies anomalous battery telemetry signatures (sensor drift, abnormal impedance growth, unexpected thermal spikes, voltage divergence) without requiring labeled failure events.
- **Algorithm:** Isolation Forest ($150$ isolation trees, $5\%$ contamination assumption).
- **Calibration:** Raw isolation tree path depths are inverted and normalized onto a continuous $[0, 100]$ Anomaly Severity Index.
- **Attribution:** Computes sensor-level $Z$-score deviations against nominal fleet baselines to highlight contributing anomalous sensors.

### 5.2 Performance & Behavior

- **Anomaly Prevalence:** ~5.0% of fleet flagged as anomalous.
- **Score Range:** $0.0$ (completely nominal) to $100.0$ (severe multivariate sensor outlier).
- **Risk Tiers:** Normal ($< 35$), Elevated ($35 - 65$), Critical ($> 65$).

---

## 6. Model 5: Composite Battery Risk Index (`battery_risk_index`)

### 6.1 Diagnostic Formulation

The composite risk index is a deterministic, multi-factor engineering diagnostic score ($0 - 100$):

$$\text{Risk Score} = 0.30 \cdot R_{\text{failure}} + 0.20 \cdot R_{\text{thermal}} + 0.15 \cdot R_{\text{electrical}} + 0.15 \cdot R_{\text{charging}} + 0.10 \cdot R_{\text{degradation}} + 0.10 \cdot R_{\text{anomaly}}$$

Where:

- $R_{\text{failure}} = P(\text{failure}) \times 100$
- $R_{\text{thermal}} = \text{clip}\left(2.2 \cdot \max(0, T_{\max} - 35) + 0.5 \cdot \max(0, 80 - \text{CoolingHealth}), 0, 100\right)$
- $R_{\text{electrical}} = \text{clip}\left(\frac{\Delta V_{\text{imbalance}}}{80\,\text{mV}} \times 100, 0, 100\right)$
- $R_{\text{charging}} = \text{clip}(\text{FastChargeRatio} \times 100, 0, 100)$
- $R_{\text{degradation}} = \text{clip}\left(\frac{100 - \text{SOH}}{30} \times 100, 0, 100\right)$
- $R_{\text{anomaly}} = \text{Anomaly Score}_{100}$

---

## 7. Operational Boundaries & Ethical Governance

### 7.1 Intended Use

- Telemetry monitoring and predictive maintenance for EV passenger fleets and commercial trucks.
- Early warning indicator for BMS diagnostics and service scheduling.
- Sensitivity and counterfactual what-if simulation for charging infrastructure optimization.

### 7.2 Non-Intended Use

- **Not a physical safety certificate:** Model outputs do not replace hardware BMS circuit breakers, pyrofuses, or physical electrochemical cell teardowns.
- **Not causal proof:** SHAP explanations quantify statistical model feature contributions and must not be interpreted as physical proof of electrochemical causation.
- **Out-of-distribution inputs:** Vehicles outside the physical boundary ranges ($T \in [-30^\circ\text{C}, 85^\circ\text{C}]$, $V \in [100\text{V}, 900\text{V}]$) should be inspected manually.
