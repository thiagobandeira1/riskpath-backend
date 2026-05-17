"""Inference + SHAP layer for the deployed V7 XGBoost readmission model.

The web app is XGBoost-only: XGBoost is the deployment-selected model, per the
revised May 2026 protocol. LightGBM scores marginally higher on the held-out
test partition (0.7960 vs 0.7929 for seed 0), but the gap is within seed-level
variance and XGBoost is retained for continuity with the defended original
capstone.

Usage:
    from src.inference import ReadmissionPredictor
    p = ReadmissionPredictor()                      # loads everything once
    proba = p.predict_proba(patient_df)             # numpy array, shape (n,)
    shap_values = p.explain(patient_df)             # dict with shap_vals, base
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .preprocess import build_encoders, transform
from .schema import FEATURE_COLS

_MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
_MODEL_FILE = "v7_seed0_xgboost.pkl"


class ReadmissionPredictor:
    """Single-entry-point class for predictions + SHAP using the deployed
    seed-0 XGBoost model."""

    def __init__(self):
        self.model_name = "xgboost"
        self.model = joblib.load(_MODEL_DIR / _MODEL_FILE)
        self.encoders = build_encoders()
        self._explainer = None  # lazy-built

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return per-row 30-day readmission probability."""
        X = transform(df, self.encoders)
        return self.model.predict_proba(X[FEATURE_COLS])[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return per-row binary readmission prediction at the given threshold."""
        return (self.predict_proba(df) >= threshold).astype(int)

    def explain(
        self,
        df: pd.DataFrame,
        *,
        max_rows: int = 100,
    ) -> dict:
        """Compute SHAP values for the given rows.

        Returns
        -------
        dict with:
            shap_values : ndarray (n, 50) — per-feature contributions for class 1
            base_value  : float — model expected value (log-odds for sklearn API)
            feature_names : list[str] of length 50, in column order
            X_transformed : DataFrame fed to the model (for visualization)
        """
        import shap  # imported lazily — only needed for explanations

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)

        X = transform(df.head(max_rows), self.encoders)[FEATURE_COLS]
        sv = self._explainer.shap_values(X)
        # XGBoost binary returns a single (n, n_features) array in modern SHAP.
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]

        base = self._explainer.expected_value
        if hasattr(base, "__len__"):
            base = base[1] if len(base) > 1 else base[0]

        return {
            "shap_values": np.asarray(sv),
            "base_value": float(base),
            "feature_names": list(FEATURE_COLS),
            "X_transformed": X,
        }
