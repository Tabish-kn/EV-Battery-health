"""
Data Quality, Integrity & Sensor Validation Pipeline.
Implements physical boundary validation, sensor spike clipping, outlier detection,
target leakage auditing, and automated data quality reporting.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.config import PHYSICAL_BOUNDS, TARGET_FAILURE, TARGET_SOH, TARGET_RUL, ID_COLUMNS


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates ingested dataset against schema, missingness, duplicate records,
    and physical plausibility ranges.
    
    Returns:
        Dictionary containing comprehensive validation results and warnings.
    """
    results = {
        "valid": True,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_summary": {},
        "boundary_violations": {},
        "warnings": [],
        "errors": []
    }
    
    if len(df) == 0:
        results["valid"] = False
        results["errors"].append("Dataset is completely empty.")
        return results

    # 1. Missingness Analysis
    missing_counts = df.isna().sum()
    for col, count in missing_counts[missing_counts > 0].items():
        pct = float(count / len(df) * 100.0)
        results["missing_summary"][col] = {
            "missing_count": int(count),
            "missing_percent": round(pct, 2)
        }
        if pct > 40.0:
            results["warnings"].append(f"High missingness in '{col}': {pct:.1f}% values missing.")

    # 2. Duplicate Records Check
    if results["duplicate_rows"] > 0:
        results["warnings"].append(f"Found {results['duplicate_rows']} exact duplicate rows.")

    # 3. Physical Boundary Violations
    for col, (min_val, max_val) in PHYSICAL_BOUNDS.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            below_min = int((df[col] < min_val).sum())
            above_max = int((df[col] > max_val).sum())
            total_violations = below_min + above_max
            if total_violations > 0:
                results["boundary_violations"][col] = {
                    "expected_range": [min_val, max_val],
                    "below_min_count": below_min,
                    "above_max_count": above_max,
                    "violation_pct": round(total_violations / len(df) * 100.0, 2)
                }

    # 4. Infinite Values Check
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols].values).sum()
    if inf_counts > 0:
        results["warnings"].append(f"Found {int(inf_counts)} infinite values across numerical columns.")

    return results


def clean_dataset(df: pd.DataFrame, clip_spikes: bool = True) -> pd.DataFrame:
    """
    Cleans telemetry data by clipping out-of-bound sensor spikes to physical boundaries
    and replacing infinities with NaN for subsequent imputer handling.
    """
    df_clean = df.copy()
    
    # Replace infinities with NaN
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        df_clean[c] = df_clean[c].replace([np.inf, -np.inf], np.nan)
        
    if clip_spikes:
        for col, (min_val, max_val) in PHYSICAL_BOUNDS.items():
            if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)
                
    return df_clean


def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    iqr_multiplier: float = 2.5
) -> Dict[str, Any]:
    """
    Identifies statistical outliers per column using IQR or Z-score thresholds.
    
    Args:
        df: Input DataFrame.
        columns: List of numerical columns to inspect (defaults to all numeric).
        method: 'iqr' or 'zscore'.
        iqr_multiplier: Multiplier for IQR bounds (2.5 is conservative for sensor spikes).
        
    Returns:
        Dictionary with outlier counts and row indices.
    """
    if columns is None:
        columns = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ID_COLUMNS]
        
    outlier_report = {}
    
    for col in columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        if method == "iqr":
            q25, q75 = series.quantile(0.25), series.quantile(0.75)
            iqr = q75 - q25
            if iqr == 0:
                continue
            lower_bound = q25 - iqr_multiplier * iqr
            upper_bound = q75 + iqr_multiplier * iqr
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        elif method == "zscore":
            std = series.std()
            if std == 0:
                continue
            z_scores = (df[col] - series.mean()) / std
            outlier_mask = z_scores.abs() > 3.5
        else:
            raise ValueError(f"Unknown outlier method '{method}'. Use 'iqr' or 'zscore'.")
            
        count = int(outlier_mask.sum())
        if count > 0:
            outlier_report[col] = {
                "outlier_count": count,
                "outlier_percentage": round(count / len(df) * 100.0, 2),
                "lower_bound": float(lower_bound) if method == "iqr" else float(series.mean() - 3.5 * series.std()),
                "upper_bound": float(upper_bound) if method == "iqr" else float(series.mean() + 3.5 * series.std()),
            }
            
    return outlier_report


def check_target_leakage(
    df: pd.DataFrame,
    target_col: str = TARGET_FAILURE,
    correlation_threshold: float = 0.95
) -> List[Dict[str, Any]]:
    """
    Audits features for prospective target leakage or artificial deterministic targets.
    
    Args:
        df: Input DataFrame.
        target_col: Column name of target.
        correlation_threshold: Pearson correlation above which a feature is flagged as leak.
        
    Returns:
        List of flagged features with correlation magnitude and warning notes.
    """
    flagged = []
    if target_col not in df.columns:
        return flagged
        
    target_series = df[target_col]
    if not pd.api.types.is_numeric_dtype(target_series):
        return flagged
        
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    
    for col in numeric_cols:
        valid_idx = df[[col, target_col]].dropna().index
        if len(valid_idx) < 10:
            continue
        corr = df.loc[valid_idx, col].corr(df.loc[valid_idx, target_col])
        if np.isnan(corr):
            continue
        if abs(corr) >= correlation_threshold:
            flagged.append({
                "feature": col,
                "target": target_col,
                "correlation": round(float(corr), 4),
                "risk": "HIGH_TARGET_LEAKAGE" if abs(corr) > 0.98 else "ELEVATED_ASSOCIATION",
                "recommendation": "Exclude feature from training matrix to prevent trivial leakage."
            })
            
    return flagged


def generate_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a structured, end-to-end data quality audit summary report.
    """
    val_res = validate_dataset(df)
    outliers = detect_outliers(df)
    
    leakage_failure = check_target_leakage(df, TARGET_FAILURE) if TARGET_FAILURE in df.columns else []
    leakage_soh = check_target_leakage(df, TARGET_SOH) if TARGET_SOH in df.columns else []
    
    report = {
        "dataset_shape": {"rows": len(df), "columns": df.shape[1]},
        "validation_passed": val_res["valid"],
        "duplicate_rows": val_res["duplicate_rows"],
        "columns_with_missing_values": len(val_res["missing_summary"]),
        "columns_with_physical_boundary_violations": len(val_res["boundary_violations"]),
        "outlier_columns_count": len(outliers),
        "target_leakage_risks": leakage_failure + leakage_soh,
        "warnings": val_res["warnings"],
        "quality_score": max(0.0, 100.0 - (len(val_res["missing_summary"]) * 0.4) - (len(val_res["boundary_violations"]) * 0.5) - (val_res["duplicate_rows"] * 0.1))
    }
    
    return report
