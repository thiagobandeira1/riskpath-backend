"""Tests for POST /explanations."""

from __future__ import annotations

import math

from fastapi.testclient import TestClient

from src.schema import FEATURE_COLS


def test_explain_returns_200(client: TestClient, valid_patient_dict: dict) -> None:
    r = client.post("/explanations", json=valid_patient_dict)
    assert r.status_code == 200, r.text


def test_explain_shape(client: TestClient, valid_patient_dict: dict) -> None:
    r = client.post("/explanations", json=valid_patient_dict)
    body = r.json()
    assert len(body["shap_values"]) == 50
    assert len(body["feature_names"]) == 50
    assert len(body["feature_values_transformed"]) == 50
    assert isinstance(body["base_value"], float)
    assert 0.0 <= body["probability"] <= 1.0


def test_explain_feature_names_in_canonical_order(
    client: TestClient, valid_patient_dict: dict
) -> None:
    r = client.post("/explanations", json=valid_patient_dict)
    assert r.json()["feature_names"] == FEATURE_COLS


def test_explain_logit_consistency(
    client: TestClient, valid_patient_dict: dict
) -> None:
    """sum(shap_values) + base_value should ~equal logit(probability).

    This is the SHAP additivity property — guards against the SHAP
    contributions being computed against a different model or row than
    what predict_proba received.
    """
    r = client.post("/explanations", json=valid_patient_dict)
    body = r.json()
    logit = body["base_value"] + sum(body["shap_values"])
    proba_from_logit = 1 / (1 + math.exp(-logit))
    # Generous tolerance accounts for float rounding through the sum;
    # SHAP for XGBoost binary classifiers is essentially exact (model_output="raw").
    assert abs(proba_from_logit - body["probability"]) < 1e-4, (
        f"additivity broken: logit={logit}, "
        f"proba_from_logit={proba_from_logit}, reported_proba={body['probability']}"
    )


def test_explain_fallback_warnings_surfaced(
    client: TestClient, valid_patient_dict: dict
) -> None:
    bad = dict(valid_patient_dict)
    bad["drg_code"] = "__UNSEEN_DRG_FOR_EXPLAIN__"
    r = client.post("/explanations", json=bad)
    assert r.status_code == 200
    body = r.json()
    assert body["fallback_warnings"]
    assert any("drg_code" in w for w in body["fallback_warnings"])


def test_explain_missing_field_returns_422(
    client: TestClient, valid_patient_dict: dict
) -> None:
    bad = {k: v for k, v in valid_patient_dict.items() if k != "bilirubin_max"}
    r = client.post("/explanations", json=bad)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_explain_response_shape(
    client: TestClient, valid_patient_dict: dict
) -> None:
    r = client.post("/explanations", json=valid_patient_dict)
    assert set(r.json().keys()) == {
        "shap_values",
        "base_value",
        "feature_names",
        "feature_values_transformed",
        "probability",
        "model_name",
        "fallback_warnings",
    }
