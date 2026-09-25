"""
Utility functions for EV Battery Intelligence Platform.
Implements batch inference pipelines, structured diagnostic report generation (HTML / JSON),
and operational helpers.
"""

from typing import Dict, Any, Optional, List
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

from src.models import ModelBundle
from src.feature_engine import get_feature_matrix, clip_sensor_spikes, engineer_physics_features
from src.risk_scoring import compute_battery_risk_score
from src.config import GENERATED_REPORTS_DIR


def run_batch_pipeline(
    df: pd.DataFrame,
    failure_bundle: ModelBundle,
    soh_bundle: Optional[ModelBundle] = None,
    rul_bundle: Optional[ModelBundle] = None,
    anomaly_bundle: Optional[ModelBundle] = None
) -> pd.DataFrame:
    """
    Executes full multi-target batch diagnostic pipeline across a fleet DataFrame.
    """
    df_out = df.copy()

    # 1. Feature Engineering
    df_clean = clip_sensor_spikes(df_out)
    df_feat = engineer_physics_features(df_clean)

    # 2. Failure Prediction
    X_fail, _, _ = get_feature_matrix(df_out, target_type="failure")
    fail_probas = failure_bundle.predict_proba(X_fail)[:, 1]
    fail_preds = (fail_probas >= 0.50).astype(int)
    df_out["predicted_failure_probability"] = np.round(fail_probas, 4)
    df_out["predicted_failure_flag"] = fail_preds

    # 3. SOH Prediction
    if soh_bundle is not None:
        X_soh, _, _ = get_feature_matrix(df_out, target_type="soh")
        soh_preds = np.clip(soh_bundle.predict(X_soh), 0.0, 100.0)
        df_out["predicted_soh_percent"] = np.round(soh_preds, 2)
    else:
        df_out["predicted_soh_percent"] = df_out.get("state_of_health", 85.0)

    # 4. RUL Prediction
    if rul_bundle is not None:
        X_rul, _, _ = get_feature_matrix(df_out, target_type="rul")
        rul_preds = np.maximum(0.0, rul_bundle.predict(X_rul))
        df_out["predicted_rul_cycles"] = np.round(rul_preds, 0)
    else:
        df_out["predicted_rul_cycles"] = df_out.get("predicted_remaining_life_cycles", 5000.0)

    # 5. Anomaly Detection
    if anomaly_bundle is not None:
        X_anom, _, _ = get_feature_matrix(df_out, target_type="anomaly")
        X_anom_scaled = anomaly_bundle.preprocessor.transform(X_anom)
        anom_scores = anomaly_bundle.estimator.calculate_anomaly_score_0_100(X_anom_scaled)
        anom_preds = anomaly_bundle.estimator.predict(X_anom_scaled)
        df_out["anomaly_score_100"] = np.round(anom_scores, 1)
        df_out["anomaly_flag"] = (anom_preds == -1).astype(int)
    else:
        df_out["anomaly_score_100"] = 15.0
        df_out["anomaly_flag"] = 0

    # 6. Risk Scoring per row
    risk_scores = []
    risk_categories = []

    for idx, row in df_feat.iterrows():
        r = compute_battery_risk_score(
            failure_probability=float(df_out.loc[idx, "predicted_failure_probability"]),
            soh_percent=float(df_out.loc[idx, "predicted_soh_percent"]),
            cell_temp_max=float(row.get("cell_temperature_max", 35.0)),
            voltage_imbalance_mv=float(row.get("voltage_imbalance", 15.0)),
            fast_charge_ratio=float(row.get("fast_charge_ratio", 0.3)),
            anomaly_score_100=float(df_out.loc[idx, "anomaly_score_100"]),
            cooling_health=float(row.get("cooling_system_health", 80.0)),
            cycle_count=float(row.get("cycle_count", 1000.0))
        )
        risk_scores.append(r["composite_risk_score"])
        risk_categories.append(r["risk_category"])

    df_out["battery_risk_score"] = risk_scores
    df_out["risk_category"] = risk_categories

    return df_out


def generate_diagnostic_html_report(
    battery_id: str,
    diagnostic_summary: Dict[str, Any],
    top_factors: List[Dict[str, Any]],
    output_filepath: Optional[str] = None
) -> str:
    """
    Generates a professional, self-contained HTML diagnostic audit certificate.
    """
    os.makedirs(GENERATED_REPORTS_DIR, exist_ok=True)
    if output_filepath is None:
        filename = f"report_{battery_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        output_filepath = os.path.join(GENERATED_REPORTS_DIR, filename)

    risk_cat = diagnostic_summary.get("risk_category", "NORMAL")
    risk_color = diagnostic_summary.get("color", "#10b981")

    factors_html = "".join([
        f"""
        <tr>
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155;">{f['feature']}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155; font-weight: 600;">{f['feature_value']}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155; color: {'#ef4444' if f['impact'] == 'INCREASES_RISK' else '#10b981'};">{f['impact']}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155;">{f['shap_value']:+.4f}</td>
        </tr>
        """
        for f in top_factors
    ])

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>EV Battery Diagnostic Report - {battery_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 20px; border: 1px solid #334155; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #334155; padding-bottom: 16px; margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px; }}
        .metric-box {{ background: #0f172a; border-radius: 8px; padding: 16px; border: 1px solid #334155; }}
        .metric-title {{ font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; }}
        .metric-val {{ font-size: 1.5rem; font-weight: bold; margin-top: 4px; color: #f8fafc; }}
        .badge {{ display: inline-block; padding: 6px 14px; border-radius: 9999px; font-weight: bold; background: {risk_color}; color: #ffffff; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.9rem; }}
        th {{ background: #0f172a; padding: 10px 12px; text-align: left; color: #94a3b8; border-bottom: 2px solid #334155; }}
        .footer {{ font-size: 0.75rem; color: #64748b; margin-top: 30px; text-align: center; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <div>
                <h1 style="margin: 0; font-size: 1.8rem; color: #38bdf8;">⚡ EV Battery Health Diagnostic Report</h1>
                <p style="margin: 4px 0 0 0; color: #94a3b8;">Pack / Vehicle ID: <strong>{battery_id}</strong> | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div>
                <span class="badge">{risk_cat} RISK</span>
            </div>
        </div>

        <div class="grid">
            <div class="metric-box">
                <div class="metric-title">Composite Risk Score</div>
                <div class="metric-val" style="color: {risk_color};">{diagnostic_summary.get('composite_risk_score', 'N/A')} / 100</div>
            </div>
            <div class="metric-box">
                <div class="metric-title">Failure Probability</div>
                <div class="metric-val">{diagnostic_summary.get('failure_probability', 0.0):.1%}</div>
            </div>
            <div class="metric-box">
                <div class="metric-title">State of Health (SOH)</div>
                <div class="metric-val">{diagnostic_summary.get('soh_percent', 0.0):.1f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-title">Remaining Useful Life</div>
                <div class="metric-val">{diagnostic_summary.get('rul_cycles', 0):,.0f} cycles</div>
            </div>
            <div class="metric-box">
                <div class="metric-title">Anomaly Index</div>
                <div class="metric-val">{diagnostic_summary.get('anomaly_score_100', 0.0):.1f} / 100</div>
            </div>
        </div>

        <h3 style="margin-top: 24px; color: #f8fafc;">Diagnostic Sub-Indices Breakdown</h3>
        <div class="grid">
            <div class="metric-box"><div class="metric-title">Thermal Risk</div><div class="metric-val" style="font-size: 1.2rem;">{diagnostic_summary.get('sub_scores', {}).get('thermal_risk', 'N/A')}/100</div></div>
            <div class="metric-box"><div class="metric-title">Electrical Risk</div><div class="metric-val" style="font-size: 1.2rem;">{diagnostic_summary.get('sub_scores', {}).get('electrical_risk', 'N/A')}/100</div></div>
            <div class="metric-box"><div class="metric-title">Charging Stress</div><div class="metric-val" style="font-size: 1.2rem;">{diagnostic_summary.get('sub_scores', {}).get('charging_risk', 'N/A')}/100</div></div>
            <div class="metric-box"><div class="metric-title">Degradation Fade</div><div class="metric-val" style="font-size: 1.2rem;">{diagnostic_summary.get('sub_scores', {}).get('degradation_risk', 'N/A')}/100</div></div>
        </div>

        <h3 style="margin-top: 24px; color: #f8fafc;">Key Diagnostic Feature Attributions (SHAP Analytics)</h3>
        <table>
            <thead>
                <tr>
                    <th>Telemetry Sensor Feature</th>
                    <th>Measured Value</th>
                    <th>Diagnostic Impact</th>
                    <th>Attribution (SHAP)</th>
                </tr>
            </thead>
            <tbody>
                {factors_html}
            </tbody>
        </table>

        <div class="footer">
            EV Battery Intelligence Platform &copy; 2026. Certified diagnostic report generated by calibrated ML & Physics Pipelines.<br>
            Note: Diagnostic risk outputs represent statistical model estimations and do not replace physical electrochemical teardown.
        </div>
    </div>
</body>
</html>
"""
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_filepath
