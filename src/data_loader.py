"""
Data Loading and Ingestion Engine for EV Battery Telemetry.
Handles CSV ingestion, sample extraction, schema integrity checks, and train/test splitting.
"""

import os
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    SAMPLE_DATA_PATH,
    PROCESSED_DATA_PATH,
    TARGET_FAILURE,
    RANDOM_STATE
)


def load_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads battery telemetry dataset from path or fallback to RAW_DATA_PATH.
    
    Args:
        file_path: Path to dataset CSV file.
        
    Returns:
        pd.DataFrame containing telemetry records.
    """
    if file_path is None:
        file_path = RAW_DATA_PATH
        if not os.path.exists(file_path):
            # Fallback to local root if data/raw is not present yet
            file_path = "ev_battery_health.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Telemetry dataset file not found at: {file_path}")

    df = pd.read_csv(file_path)
    return df


def load_raw_data() -> pd.DataFrame:
    """Convenience helper to load raw benchmark telemetry."""
    return load_data(str(RAW_DATA_PATH))


def get_sample_data(n_samples: int = 50) -> pd.DataFrame:
    """
    Loads or generates a curated sample batch for demonstration, what-if, or UI testing.
    """
    if os.path.exists(SAMPLE_DATA_PATH):
        df_sample = pd.read_csv(SAMPLE_DATA_PATH)
        return df_sample.head(n_samples)
    
    df = load_raw_data()
    return df.head(n_samples)


def save_processed_data(df: pd.DataFrame, output_path: Optional[str] = None) -> str:
    """
    Saves transformed / feature-engineered dataframe to CSV.
    """
    if output_path is None:
        output_path = str(PROCESSED_DATA_PATH)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_FAILURE,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
    stratify_target: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs train/test split with optional stratification for binary targets.
    
    Args:
        df: Input DataFrame with features and target.
        target_col: Name of target column.
        test_size: Fraction of test split.
        random_state: Random state seed.
        stratify_target: Whether to stratify split by target values.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    
    # Filter rows where target is NaN
    valid_mask = df[target_col].notna()
    df_clean = df[valid_mask].copy()
    
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    stratify = y if (stratify_target and y.nunique() <= 10) else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    return X_train, X_test, y_train, y_test
