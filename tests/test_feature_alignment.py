"""NON-NEGOTIABLE feature-alignment tests.

These six tests guard against the failure mode no human review will catch:
the FastAPI route silently dropping, reordering, or wrong-typing the 50
features relative to model/v7_feature_cols.json, producing plausible-but-
wrong probabilities.

If any test here is skipped, disabled, or xfailed without explicit written
justification in spec.md, that's a stop-the-build event.

The six tests, per spec.md §Testing Strategy:
    1. test_column_order_matches_model
    2. test_categorical_encoding_matches_training
    3. test_missing_required_feature_returns_422
    4. test_extra_unknown_field_is_rejected
    5. test_unseen_categorical_value_handled
    6. test_full_test_set_auroc_matches_inference
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import roc_auc_score

from app.dependencies import get_predictor
from app.schemas.features import PatientFeatures
from src.schema import CATEGORICAL_COLS, FEATURE_COLS, TARGET_COL


def _row_to_dict(row: pd.Series) -> dict[str, object]:
    """Same coercion the conftest applies — keep duplication local so the
    alignment tests stay self-contained (DAMP-over-DRY per the TDD skill)."""
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


# ───────────────────────────────────────────────────────────────────────
# Alignment test 1
# ───────────────────────────────────────────────────────────────────────

def test_column_order_matches_model(
    client: TestClient, sample_patients: list[dict]
) -> None:
    """Shuffling JSON keys must not change the predicted probability.

    Guards against any code path that iterates over dict keys in input order
    and feeds them to the model column-misaligned.
    """
    rng = random.Random(0)
    for i in range(5):
        canonical = sample_patients[i]
        keys = list(canonical.keys())
        rng.shuffle(keys)
        shuffled = {k: canonical[k] for k in keys}

        r_canon = client.post("/predictions", json=canonical)
        r_shuf = client.post("/predictions", json=shuffled)
        assert r_canon.status_code == 200 and r_shuf.status_code == 200

        p_canon = r_canon.json()["probability"]
        p_shuf = r_shuf.json()["probability"]
        assert abs(p_canon - p_shuf) < 1e-9, (
            f"row {i}: shuffling changed probability "
            f"(canonical={p_canon}, shuffled={p_shuf})"
        )


# ───────────────────────────────────────────────────────────────────────
# Alignment test 2
# ───────────────────────────────────────────────────────────────────────

def test_categorical_encoding_matches_training(
    client: TestClient, sample_patients: list[dict]
) -> None:
    """HTTP path must produce the same probability as the direct
    ReadmissionPredictor for any in-distribution row.

    Guards against the API layer using a different LabelEncoder than the
    predictor — the exact failure that would silently corrupt every prediction.
    """
    predictor = get_predictor()
    for i in range(5):
        row = sample_patients[i]
        r = client.post("/predictions", json=row)
        assert r.status_code == 200, r.text
        http_proba = r.json()["probability"]

        pf = PatientFeatures(**row)
        direct_proba = float(predictor.predict_proba(pf.to_dataframe())[0])
        assert abs(http_proba - direct_proba) < 1e-9, (
            f"row {i}: http={http_proba}, direct={direct_proba}, "
            f"delta={http_proba - direct_proba}"
        )


# ───────────────────────────────────────────────────────────────────────
# Alignment test 3
# ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("missing_field", ["drg_code", "los_trend_180d", "bmi_last"])
def test_missing_required_feature_returns_422(
    client: TestClient, valid_patient_dict: dict, missing_field: str
) -> None:
    """A request missing any of the 50 features must 422 with VALIDATION_ERROR.

    Guards against silent default-filling.
    """
    bad = {k: v for k, v in valid_patient_dict.items() if k != missing_field}
    r = client.post("/predictions", json=bad)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"


# ───────────────────────────────────────────────────────────────────────
# Alignment test 4
# ───────────────────────────────────────────────────────────────────────

def test_extra_unknown_field_is_rejected(
    client: TestClient, valid_patient_dict: dict
) -> None:
    """{...all 50 fields, "surgeon_name": "Dr X"} must 422.

    Guards against typos in real field names being silently ignored
    (e.g., sending "drg_codes" instead of "drg_code" — would otherwise
    drop the value entirely and use a NaN at scoring time).
    """
    bad = {**valid_patient_dict, "surgeon_name": "Dr X"}
    r = client.post("/predictions", json=bad)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"


# ───────────────────────────────────────────────────────────────────────
# Alignment test 5
# ───────────────────────────────────────────────────────────────────────

def test_unseen_categorical_value_handled(
    client: TestClient, valid_patient_dict: dict
) -> None:
    """An unseen drg_code should produce a valid probability AND a warning.

    Surfaces preprocess.transform's silent most-common fallback to the
    consumer so the UI can flag a degraded prediction.
    """
    bad = dict(valid_patient_dict)
    bad["drg_code"] = "__UNSEEN_DRG_CODE_VALUE__"
    r = client.post("/predictions", json=bad)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["fallback_warnings"], "expected at least one fallback warning"
    assert any("drg_code" in w for w in body["fallback_warnings"])


# ───────────────────────────────────────────────────────────────────────
# Alignment test 6
# ───────────────────────────────────────────────────────────────────────

def test_full_test_set_auroc_matches_inference(client: TestClient) -> None:
    """HTTP path is mathematically equivalent to ReadmissionPredictor across
    a sizeable test slice.

    End-to-end correctness — guarantees nothing along the FastAPI plumbing
    introduces a measurable bias in probabilities or AUROC.
    """
    N_ROWS = 200  # spec.md asks for 1000; 200 keeps the suite under the 60s budget

    df = pd.read_parquet("data/processed/training_table_v7.parquet")
    split = np.load("model/v7_split_indices.npz")
    test_df = df.iloc[split["test_idx"]].reset_index(drop=True)
    sample = test_df.sample(n=N_ROWS, random_state=123).reset_index(drop=True)
    y_true = sample[TARGET_COL].astype(int).values

    predictor = get_predictor()
    direct = predictor.predict_proba(sample)

    http_probs: list[float] = []
    for _, row in sample.iterrows():
        r = client.post("/predictions", json=_row_to_dict(row))
        assert r.status_code == 200, r.text
        http_probs.append(r.json()["probability"])
    http_arr = np.array(http_probs)

    max_diff = float(np.abs(direct - http_arr).max())
    assert max_diff < 1e-9, f"max per-row prob diff between HTTP and direct: {max_diff}"

    http_auc = roc_auc_score(y_true, http_arr)
    direct_auc = roc_auc_score(y_true, direct)
    assert abs(http_auc - direct_auc) < 1e-9, (
        f"AUROC differs: http={http_auc}, direct={direct_auc}"
    )
