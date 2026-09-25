"""
Unit tests for data preprocessing and transformer pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from src.preprocessing import BatteryPreprocessor


def test_preprocessor_fit_transform():
    df = pd.DataFrame({
        "cell_voltage_avg": [3.7, 3.8, np.nan, 3.6],
        "cell_temperature_max": [35.0, 42.0, 30.0, np.nan],
        "battery_chemistry": ["NMC", "LFP", "NMC", np.nan],
        "drive_type": ["AWD", "FWD", "RWD", "AWD"]
    })
    
    preprocessor = BatteryPreprocessor(cat_cols=["battery_chemistry", "drive_type"])
    X_scaled = preprocessor.fit_transform(df)
    
    assert X_scaled.shape[0] == 4
    assert not np.isnan(X_scaled).any()
    assert preprocessor.is_fitted


def test_preprocessor_unseen_categories():
    df_train = pd.DataFrame({
        "cell_voltage_avg": [3.7, 3.8, 3.9],
        "battery_chemistry": ["NMC", "LFP", "NMC"]
    })
    df_test = pd.DataFrame({
        "cell_voltage_avg": [3.6],
        "battery_chemistry": ["UNKNOWN_CHEMISTRY"]
    })
    
    preprocessor = BatteryPreprocessor(cat_cols=["battery_chemistry"])
    preprocessor.fit(df_train)
    X_test_scaled = preprocessor.transform(df_test)
    
    assert X_test_scaled.shape[0] == 1
    assert not np.isnan(X_test_scaled).any()
