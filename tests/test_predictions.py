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


# ───────────────────────────────────────────────────────────────────────
# POST /predictions/batch
# ───────────────────────────────────────────────────────────────────────

def test_batch_happy_path(client: TestClient, sample_patients: list[dict]) -> None:
    body = {"patients": sample_patients[:5]}
    r = client.post("/predictions/batch", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert len(out["predictions"]) == 5
    for p in out["predictions"]:
        assert 0.0 <= p["probability"] <= 1.0
        assert p["prediction"] in (0, 1)


def test_batch_matches_individual_calls(
    client: TestClient, sample_patients: list[dict]
) -> None:
    """Batch of 5 returns same probabilities as 5 individual calls (1e-9).

    This is the batch-path analog of the alignment tests — guards against
    multi-row predict producing different numerics from N single-row predicts.
    """
    five = sample_patients[:5]
    individual = [
        client.post("/predictions", json=p).json()["probability"] for p in five
    ]
    batched_resp = client.post("/predictions/batch", json={"patients": five})
    assert batched_resp.status_code == 200
    batched = [p["probability"] for p in batched_resp.json()["predictions"]]

    assert len(individual) == len(batched)
    for i, (a, b) in enumerate(zip(individual, batched)):
        assert abs(a - b) < 1e-9, f"row {i}: individual={a}, batch={b}"


def test_batch_preserves_input_order(
    client: TestClient, sample_patients: list[dict]
) -> None:
    """The Nth batch response corresponds to the Nth input patient."""
    five = sample_patients[:5]
    r = client.post("/predictions/batch", json={"patients": five})
    batched = [p["probability"] for p in r.json()["predictions"]]
    individual = [
        client.post("/predictions", json=p).json()["probability"] for p in five
    ]
    assert batched == individual


def test_batch_too_large_returns_422(
    client: TestClient, sample_patients: list[dict]
) -> None:
    body = {"patients": [sample_patients[0]] * 101}
    r = client.post("/predictions/batch", json=body)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_batch_empty_returns_422(client: TestClient) -> None:
    r = client.post("/predictions/batch", json={"patients": []})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_batch_at_max_size_succeeds(
    client: TestClient, sample_patients: list[dict]
) -> None:
    """Edge case: exactly 100 patients (the cap) is accepted."""
    # Cycle the 20 sampled rows up to 100 patients
    patients = [sample_patients[i % 20] for i in range(100)]
    r = client.post("/predictions/batch", json={"patients": patients})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 100
