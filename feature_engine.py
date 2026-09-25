"""
Top-level feature_engine wrapper for backward compatibility.
Exposes all physics feature derivations, spike clippers, and alias helpers from src.feature_engine.
"""

from src.feature_engine import (
    COLUMN_ALIASES,
    PHYSICS_FEATURES,
    get_series,
    clip_sensor_spikes,
    engineer_physics_features,
    get_feature_matrix
)

__all__ = [
    'COLUMN_ALIASES',
    'PHYSICS_FEATURES',
    'get_series',
    'clip_sensor_spikes',
    'engineer_physics_features',
    'get_feature_matrix'
]
