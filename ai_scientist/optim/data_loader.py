from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

# Import from constellaration
try:
    from constellaration.generative_model.bootstrap_dataset import (
        _unflatten_metrics_and_concatenate,
        _unserialize_surface,
        load_source_datasets_with_no_errors,
    )
except ImportError:
    # Fallback or error if constellaration is not installed/found
    raise ImportError("constellaration package is required for data loading.")


class LogRobustScaler(BaseEstimator, TransformerMixin):
    """
    Custom Scaler that applies SymLog (sign(x) * log1p(|x|))
    and then applies RobustScaler (median/IQR).

    Useful for heavy-tailed physics metrics like W_MHD, QI.
    """

    def __init__(self, with_centering=True, with_scaling=True):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.robust_scaler = RobustScaler(
            with_centering=with_centering, with_scaling=with_scaling
        )
        self.is_fitted = False

    def fit(self, X, y=None):
        # SymLog transformation
        X_trans = np.sign(X) * np.log1p(np.abs(X))
        self.robust_scaler.fit(X_trans)
        self.is_fitted = True
        return self

    def transform(self, X):
        X_trans = np.sign(X) * np.log1p(np.abs(X))
        return self.robust_scaler.transform(X_trans)

    def inverse_transform(self, X):
        X_unscaled = self.robust_scaler.inverse_transform(X)
        # Inverse of sign(x)*log1p(|x|) is sign(y)*(exp(|y|) - 1)
        return np.sign(X_unscaled) * (np.expm1(np.abs(X_unscaled)))


def load_constellaration_dataset(
    filter_geometry: bool = True,
) -> pd.DataFrame:
    """
    Load the Constellaration dataset, unflatten metrics, and optionally filter by geometry validity.
    """
    print("Loading source datasets (this may take a moment)...")
    df = load_source_datasets_with_no_errors()

    print("Unflattening metrics...")
    df = _unflatten_metrics_and_concatenate(df)

    # Clean column names (remove "metrics." prefix)
    df.columns = [
        c.replace("metrics.", "") if c != "metrics.id" else c for c in df.columns
    ]

    # Remove duplicate columns resulting from renaming
    df = df.loc[:, ~df.columns.duplicated()]

    print("Unserializing surfaces...")
    df = _unserialize_surface(df)

    if filter_geometry:
        print("Filtering geometric validity...")
        df = _filter_geometric_validity(df)

    return df


def _filter_geometric_validity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows based on geometric sanity checks.
    """
    initial_len = len(df)

    # 1. Drop NaNs in critical geometric metrics
    # These columns should be present after unflattening
    critical_cols = ["max_elongation", "aspect_ratio"]
    for col in critical_cols:
        if col in df.columns:
            df = df.dropna(subset=[col])

    # 2. Filter extreme/unphysical values
    # Aspect Ratio < 1 is physically impossible for a torus (R/a)
    if "aspect_ratio" in df.columns:
        df = df[df["aspect_ratio"] >= 1.0]
        df = df[df["aspect_ratio"] < 100.0]  # Upper bound to remove outliers

    # Elongation >= 1
    if "max_elongation" in df.columns:
        df = df[df["max_elongation"] >= 1.0]
        df = df[df["max_elongation"] < 50.0]

    print(f"Filtered {initial_len - len(df)} rows based on geometric bounds.")
    return df


def save_scaler(scaler: LogRobustScaler, path: str):
    """Save the fitted scaler."""
    joblib.dump(scaler, path)


def load_scaler(path: str) -> LogRobustScaler:
    """Load a fitted scaler."""
    return joblib.load(path)
