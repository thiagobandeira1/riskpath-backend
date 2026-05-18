"""GET /examples?n=N — anonymized patient rows for one-click demo loading.

Belt-and-braces PHI defense:
    1. read_parquet(..., columns=FEATURE_COLS) — ID_COLS never enter process
       memory in the first place.
    2. Round-trip through PatientFeatures(extra="forbid") before serializing,
       so any accidental extra key would 500 the request rather than leak.
    3. test_examples.py asserts ID_COLS keys never appear in any response.

Both layer 1 and layer 3 are independently sufficient; running both is
intentional for an evidence-artefact codebase.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Query

from app.error_handlers import ERROR_RESPONSES
from app.schemas import ExamplesResponse, PatientFeatures
from src.schema import CATEGORICAL_COLS, FEATURE_COLS

router = APIRouter(tags=["examples"])

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_PARQUET = _PROJECT_ROOT / "data" / "processed" / "training_table_v7.parquet"
# Pre-baked pool of 100 anonymized rows (FEATURE_COLS only). Created by
# scripts/freeze_deploy_artifacts.py. When this pickle exists, /examples
# samples from it instead of reading the 25 MB parquet — keeps the deployed
# image slim and avoids shipping the full MIMIC-derived training table.
_DEPLOY_POOL = _PROJECT_ROOT / "model" / "deploy_examples_pool.pkl"

_MAX_N = 100


@lru_cache(maxsize=1)
def _load_examples_pool() -> pd.DataFrame:
    """Load FEATURE_COLS-only rows; ID_COLS never read into memory.

    Prefers the pre-baked pool pickle (deploy path); falls back to the
    parquet if the pickle is absent (local-dev path).
    """
    if _DEPLOY_POOL.exists():
        return pd.read_pickle(_DEPLOY_POOL)
    return pd.read_parquet(_TRAINING_PARQUET, columns=FEATURE_COLS)


def _row_to_patient_features(row: pd.Series) -> PatientFeatures:
    """Convert one parquet row to a Pydantic-validated PatientFeatures."""
    payload: dict[str, object] = {}
    for c in FEATURE_COLS:
        v = row[c]
        if pd.isna(v):
            payload[c] = None
        elif c in CATEGORICAL_COLS:
            payload[c] = str(v)
        else:
            payload[c] = float(v)
    return PatientFeatures(**payload)


@router.get(
    "/examples",
    response_model=ExamplesResponse,
    responses=ERROR_RESPONSES,
    summary="Anonymized example patient rows for demo loading",
    description=(
        "Returns N anonymized rows drawn from the local V7 parquet, each "
        "shaped for direct POST to /predictions or /explanations. All 5 "
        "ID_COLS (subject_id, hadm_id, admittime_dt, dischtime_dt, "
        "insurance) are stripped — never read into process memory in the "
        "first place. Bounded N <= 100."
    ),
)
def examples(
    n: Annotated[
        int,
        Query(
            ge=1,
            le=_MAX_N,
            description=f"Number of example rows to return (1-{_MAX_N}).",
        ),
    ] = 5,
) -> ExamplesResponse:
    pool = _load_examples_pool()
    sample = pool.sample(n=n, random_state=42).reset_index(drop=True)
    examples_list = [_row_to_patient_features(sample.iloc[i]) for i in range(len(sample))]
    return ExamplesResponse(examples=examples_list, n=n)
