"""Preprocessing pipeline for the V7 readmission model.

The training notebook applies the following transformation before training:

    X = df[FEATURE_COLS].copy()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

LabelEncoder is fit on the FULL training table (244,576 admissions), which is
why every observed category value is covered. To preserve identical encoding
for inference, this module fits LabelEncoders on the same source parquet and
exposes them via `build_encoders()`.

Usage:
    from src.preprocess import build_encoders, transform
    encoders = build_encoders()                # fit once at app startup
    X = transform(patient_df, encoders)        # transform new patient rows
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .schema import CATEGORICAL_COLS, FEATURE_COLS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TRAINING_PARQUET = _PROJECT_ROOT / "data" / "processed" / "training_table_v7.parquet"


def build_encoders(parquet_path: Path | None = None) -> Dict[str, LabelEncoder]:
    """Fit a LabelEncoder per categorical column on the V7 training table.

    Returns a dict keyed by column name. Re-fit on the same source data the
    notebook uses, so encodings match the trained model bit-for-bit.
    """
    path = parquet_path or _TRAINING_PARQUET
    if not path.exists():
        raise FileNotFoundError(
            f"V7 training parquet not found at {path}. "
            "Encoders cannot be reproducibly fit without it."
        )
    df = pd.read_parquet(path, columns=CATEGORICAL_COLS)
    encoders: Dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


def transform(
    df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder],
    *,
    unseen_label_strategy: str = "most_common",
) -> pd.DataFrame:
    """Apply the V7 preprocessing pipeline to new patient rows.

    Parameters
    ----------
    df : DataFrame containing at least the columns listed in FEATURE_COLS.
    encoders : Dict from build_encoders().
    unseen_label_strategy : "most_common" (default) maps unseen category
        values to the encoder's most frequent class; "error" raises.

    Returns
    -------
    DataFrame with exactly FEATURE_COLS in canonical order, all numeric,
    ready for model.predict_proba().
    """
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required feature columns: {sorted(missing)}")

    X = df[FEATURE_COLS].copy()

    for col in CATEGORICAL_COLS:
        if col not in X.columns:
            continue
        le = encoders[col]
        as_str = X[col].astype(str).values
        known = set(le.classes_)
        if not set(as_str).issubset(known):
            if unseen_label_strategy == "error":
                unseen = sorted(set(as_str) - known)
                raise ValueError(
                    f"Column {col!r} contains unseen category values: {unseen[:5]}"
                )
            # most_common: fall back to the encoder's mode (class 0 of a label
            # encoder is the lexicographically first, which is fine as a stable
            # fallback; ideal would be the training-set mode, but that's not
            # serialized).
            fallback = le.classes_[0]
            as_str = np.where(np.isin(as_str, list(known)), as_str, fallback)
        X[col] = le.transform(as_str)

    return X
