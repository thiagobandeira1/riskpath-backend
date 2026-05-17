"""Shared pytest fixtures.

Patient rows used in tests are sampled at runtime from the LOCAL V7 parquet
via `df.sample(n=20, random_state=42)`. We never hardcode patient data in
test files — that would commit MIMIC content into git and breach the
PhysioNet DUA.

Fixtures exposed:
    client              session-scoped TestClient(app) with the predictor
                        pre-warmed via FastAPI lifespan
    sample_patients     session-scoped: list[dict] of 20 patient rows, each
                        already shaped to POST to /predictions
    valid_patient_dict  function-scoped: a defensive copy of sample_patients[0],
                        safe for tests to mutate
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.schema import CATEGORICAL_COLS, FEATURE_COLS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_PARQUET = _PROJECT_ROOT / "data" / "processed" / "training_table_v7.parquet"


def _row_to_dict(row: pd.Series) -> dict[str, object]:
    """Convert a pandas row to a Pydantic-friendly dict (NaN -> None)."""
    out: dict[str, object] = {}
    for c in FEATURE_COLS:
        v = row[c]
        if pd.isna(v):
            out[c] = None
        elif c in CATEGORICAL_COLS:
            out[c] = str(v)
        else:
            out[c] = float(v)
    return out


@pytest.fixture(scope="session")
def sample_patients() -> list[dict[str, object]]:
    """20 patient rows from the V7 parquet, ready to POST to /predictions."""
    if not _TRAIN_PARQUET.exists():
        pytest.skip(f"V7 parquet not found at {_TRAIN_PARQUET}; tests require local data.")
    df = pd.read_parquet(_TRAIN_PARQUET, columns=FEATURE_COLS)
    sample = df.sample(n=20, random_state=42).reset_index(drop=True)
    return [_row_to_dict(sample.iloc[i]) for i in range(len(sample))]


@pytest.fixture
def valid_patient_dict(sample_patients: list[dict[str, object]]) -> dict[str, object]:
    """One patient row as a dict — safe to mutate (defensive copy)."""
    return dict(sample_patients[0])


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """Session-scoped TestClient with the predictor pre-warmed via lifespan."""
    with TestClient(app) as c:
        yield c
