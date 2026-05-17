"""V7 feature schema for the readmission prediction model.

The model was trained on 50 features (the "V7 parsimonious set") with patient-
grouped 80/20 train-test split + 10% inner-validation slice for early stopping.

This module exposes:
    FEATURE_COLS       — ordered list of the 50 feature column names
    CATEGORICAL_COLS   — subset of FEATURE_COLS that are object/string-valued
    NUMERIC_COLS       — subset that are numeric
    ID_COLS            — columns present in the parquet but NOT used as features
    TARGET_COL         — name of the binary outcome column

Load with: from src.schema import FEATURE_COLS, CATEGORICAL_COLS
"""

from __future__ import annotations

import json
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

# 50 features in canonical training order. Loaded once at import time so the
# rest of the app can reference FEATURE_COLS without re-reading JSON.
FEATURE_COLS: list[str] = json.loads(
    (_MODEL_DIR / "v7_feature_cols.json").read_text()
)

# Columns present in training_table_v7.parquet that are NOT model features.
ID_COLS: list[str] = [
    "subject_id", "hadm_id", "admittime_dt", "dischtime_dt", "insurance",
]

TARGET_COL: str = "readmit_30d"

# Categorical features (object dtype in the source parquet). These need
# LabelEncoder-style integer encoding before being passed to the model.
# (Determined by inspecting the V7 parquet dtypes at audit time.)
CATEGORICAL_COLS: list[str] = [
    "drg_code", "last_drg_dispo", "discharge_location", "primary_dx_chapter",
]

NUMERIC_COLS: list[str] = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]

assert len(FEATURE_COLS) == 50, (
    f"Expected 50 features, got {len(FEATURE_COLS)}. "
    "Check model/v7_feature_cols.json."
)
