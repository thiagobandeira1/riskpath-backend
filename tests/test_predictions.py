"""Tests for POST /predictions (single patient).

Batch + explanations live in tests/test_predictions.py extensions (Task 8)
and tests/test_explanations.py (Task 9). Alignment-critical tests live in
tests/test_feature_alignment.py.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_predict_happy_path(client: TestClient, valid_patient_dict: dict) -> None:
    r = client.post("/predictions", json=valid_patient_dict)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["threshold"] == 0.5
    assert body["model_name"] == "xgboost-v7-seed0"
    assert body["fallback_warnings"] == []


def test_predict_default_threshold_is_half(client: TestClient, valid_patient_dict: dict) -> None:
    r = client.post("/predictions", json=valid_patient_dict)
    assert r.json()["threshold"] == 0.5


def test_predict_custom_threshold_flips_prediction(
    client: TestClient, sample_patients: list[dict]
) -> None:
    """At threshold 0.0 everything is classified positive; at 1.0 everything negative."""
    row = sample_patients[0]
    r_zero = client.post("/predictions?threshold=0.0", json=row)
    r_one = client.post("/predictions?threshold=1.0", json=row)
    assert r_zero.status_code == 200
    assert r_one.status_code == 200
    # probability is the same either way; only prediction changes
    assert r_zero.json()["probability"] == r_one.json()["probability"]
    assert r_zero.json()["prediction"] == 1
    assert r_one.json()["prediction"] == 0


def test_predict_threshold_out_of_range_returns_422(
    client: TestClient, valid_patient_dict: dict
) -> None:
    r = client.post("/predictions?threshold=1.5", json=valid_patient_dict)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_predict_returns_consistent_response_shape(
    client: TestClient, valid_patient_dict: dict
) -> None:
    r = client.post("/predictions", json=valid_patient_dict)
    body = r.json()
    assert set(body.keys()) == {
        "probability", "prediction", "threshold", "model_name", "fallback_warnings"
    }
