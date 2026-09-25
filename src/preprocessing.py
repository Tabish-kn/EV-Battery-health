"""
Preprocessing Pipeline for EV Battery Telemetry.
Handles numeric imputation, categorical encoding, feature scaling,
and column consistency tracking.
"""

from typing import List, Optional, Tuple, Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLUMNS


class BatteryPreprocessor(BaseEstimator, TransformerMixin):
    """
    Standardized Battery Preprocessing Pipeline.
    Fits on train data and deterministically transforms raw telemetry dataframes.
    """

    def __init__(self, cat_cols: Optional[List[str]] = None):
        self.cat_cols = cat_cols if cat_cols is not None else list(CATEGORICAL_COLUMNS)
        self.num_cols: List[str] = []
        self.active_cat_cols: List[str] = []
        self.all_feature_names: List[str] = []
        
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit imputers, encoder, and scaler on training DataFrame."""
        X_df = X.copy()
        
        self.active_cat_cols = [c for c in self.cat_cols if c in X_df.columns]
        self.num_cols = [
            c for c in X_df.columns
            if c not in self.active_cat_cols and pd.api.types.is_numeric_dtype(X_df[c])
        ]

        # 1. Fit Numeric Imputer
        if self.num_cols:
            self.num_imputer.fit(X_df[self.num_cols])

        # 2. Fit Categorical Imputer & Encoder
        if self.active_cat_cols:
            cat_imp = self.cat_imputer.fit_transform(X_df[self.active_cat_cols].astype(str))
            self.encoder.fit(cat_imp)
            cat_encoded_names = self.encoder.get_feature_names_out(self.active_cat_cols).tolist()
        else:
            cat_encoded_names = []

        self.all_feature_names = self.num_cols + cat_encoded_names

        # 3. Fit Scaler on combined imputed array
        X_transformed = self._transform_without_scaling(X_df)
        self.scaler.fit(X_transformed)

        self.is_fitted = True
        return self

    def _transform_without_scaling(self, X: pd.DataFrame) -> np.ndarray:
        """Internal helper to apply imputation and encoding before scaling."""
        X_df = X.copy()

        # Numeric Imputation
        if self.num_cols:
            # Ensure all expected numeric columns exist in input
            for col in self.num_cols:
                if col not in X_df.columns:
                    X_df[col] = np.nan
            num_imp = self.num_imputer.transform(X_df[self.num_cols])
        else:
            num_imp = np.empty((len(X_df), 0))

        # Categorical Imputation & Encoding
        if self.active_cat_cols:
            for col in self.active_cat_cols:
                if col not in X_df.columns:
                    X_df[col] = "Missing"
            cat_imp = self.cat_imputer.transform(X_df[self.active_cat_cols].astype(str))
            cat_encoded = self.encoder.transform(cat_imp)
        else:
            cat_encoded = np.empty((len(X_df), 0))

        return np.hstack([num_imp, cat_encoded])

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform input DataFrame into scaled feature array."""
        if not self.is_fitted:
            raise RuntimeError("BatteryPreprocessor must be fitted before calling transform().")
        X_combined = self._transform_without_scaling(X)
        return self.scaler.transform(X_combined)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> List[str]:
        return list(self.all_feature_names)

    def save(self, filepath: str):
        """Serialize preprocessor to disk."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "BatteryPreprocessor":
        """Load serialized preprocessor from disk."""
        return joblib.load(filepath)
