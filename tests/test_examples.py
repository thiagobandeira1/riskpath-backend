"""Tests for GET /examples.

Includes the NON-NEGOTIABLE PHI guard from spec.md: no ID_COL key may
appear in any /examples response. Failure here is a stop-the-build
event for the same reason the alignment tests are: silent PHI leakage
would breach the PhysioNet DUA without the test surfacing it.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.schema import FEATURE_COLS, ID_COLS


def test_examples_default_n_is_5(client: TestClient) -> None:
    r = client.get("/examples")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 5
    assert len(body["examples"]) == 5


@pytest.mark.parametrize("n", [1, 5, 50, 100])
def test_examples_custom_n(client: TestClient, n: int) -> None:
    r = client.get(f"/examples?n={n}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] == n
    assert len(body["examples"]) == n


def test_examples_each_row_has_all_50_features(client: TestClient) -> None:
    r = client.get("/examples?n=5")
    body = r.json()
    for i, ex in enumerate(body["examples"]):
        assert set(ex.keys()) == set(FEATURE_COLS), (
            f"example[{i}] keys diverge from FEATURE_COLS"
        )


# ───────────────────────────────────────────────────────────────────────
# PHI GUARD — non-negotiable
# ───────────────────────────────────────────────────────────────────────

def test_examples_no_id_cols_as_keys_in_any_row(client: TestClient) -> None:
    """No ID_COL key may appear in any example object. PHI GUARD."""
    r = client.get("/examples?n=10")
    body = r.json()
    for i, ex in enumerate(body["examples"]):
        keys = set(ex.keys())
        for id_col in ID_COLS:
            assert id_col not in keys, (
                f"PHI leak: {id_col!r} appears as a key in example[{i}]"
            )


def test_examples_no_id_col_strings_in_response_body(client: TestClient) -> None:
    """Belt-and-braces: ID_COL names should not appear anywhere in the JSON.

    This catches the case where ID_COLS leak into a value (e.g., a copy-pasted
    column-name string) rather than as a key.
    """
    r = client.get("/examples?n=10")
    body_str = json.dumps(r.json())
    for id_col in ID_COLS:
        assert id_col not in body_str, (
            f"PHI leak: {id_col!r} appears somewhere in /examples response body"
        )


# ───────────────────────────────────────────────────────────────────────
# Validation
# ───────────────────────────────────────────────────────────────────────

def test_examples_n_too_large_returns_422(client: TestClient) -> None:
    r = client.get("/examples?n=101")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_examples_n_zero_returns_422(client: TestClient) -> None:
    r = client.get("/examples?n=0")
    assert r.status_code == 422


def test_examples_n_negative_returns_422(client: TestClient) -> None:
    r = client.get("/examples?n=-5")
    assert r.status_code == 422


# ───────────────────────────────────────────────────────────────────────
# End-to-end usability
# ───────────────────────────────────────────────────────────────────────

def test_example_row_can_be_posted_to_predictions(client: TestClient) -> None:
    """An /examples row should POST to /predictions cleanly — that's the
    whole point of the endpoint."""
    r = client.get("/examples?n=1")
    example = r.json()["examples"][0]
    p = client.post("/predictions", json=example)
    assert p.status_code == 200, p.text
    assert 0.0 <= p.json()["probability"] <= 1.0


def test_example_row_can_be_posted_to_explanations(client: TestClient) -> None:
    """An /examples row should POST to /explanations cleanly."""
    r = client.get("/examples?n=1")
    example = r.json()["examples"][0]
    e = client.post("/explanations", json=example)
    assert e.status_code == 200, e.text
    assert len(e.json()["shap_values"]) == 50
