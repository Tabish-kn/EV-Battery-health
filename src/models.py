"""
Base Model Registry, Serialization & Artifact Management for EV Battery Intelligence.
Provides unified ModelBundle abstraction for storing models, preprocessors, feature metadata,
and evaluation metrics.
"""

import os
import json
import joblib
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd


class ModelBundle:
    """
    Encapsulates a trained estimator, preprocessor, feature list, and metadata.
    Ensures that inference uses identical preprocessing and column alignments as training.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str,  # 'classification', 'regression', 'anomaly'
        estimator: Any,
        preprocessor: Any,
        feature_names: List[str],
        metrics: Optional[Dict[str, Any]] = None,
        version: str = "2.0.0",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.metrics = metrics or {}
        self.version = version
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow().isoformat()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Applies preprocessor and predicts target."""
        X_scaled = self.preprocessor.transform(X)
        return self.estimator.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Applies preprocessor and outputs class probabilities."""
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"Estimator '{self.model_name}' does not support predict_proba.")
        X_scaled = self.preprocessor.transform(X)
        return self.estimator.predict_proba(X_scaled)

    def save(self, directory: str, filename_prefix: Optional[str] = None):
        """
        Saves bundle binary (.pkl) and metadata (.json) to disk.
        """
        os.makedirs(directory, exist_ok=True)
        prefix = filename_prefix or self.model_name
        
        bundle_path = os.path.join(directory, f"{prefix}_bundle.pkl")
        metadata_path = os.path.join(directory, f"{prefix}_metadata.json")
        
        # Save bundle
        joblib.dump(self, bundle_path)
        
        # Save JSON metadata (serializable parts)
        meta_dict = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "created_at": self.created_at,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "metadata": self.metadata
        }
        with open(metadata_path, "w") as f:
            json.dump(meta_dict, f, indent=2)
            
        return bundle_path

    @classmethod
    def load(cls, filepath_or_dir: str) -> "ModelBundle":
        """Loads serialized ModelBundle from .pkl file or directory."""
        if os.path.isdir(filepath_or_dir):
            pkl_files = [f for f in os.listdir(filepath_or_dir) if f.endswith("_bundle.pkl") or f.endswith(".pkl")]
            if not pkl_files:
                raise FileNotFoundError(f"No pickle bundle found in directory: {filepath_or_dir}")
            filepath = os.path.join(filepath_or_dir, pkl_files[0])
        else:
            filepath = filepath_or_dir

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ModelBundle file not found: {filepath}")

        bundle = joblib.load(filepath)
        return bundle
