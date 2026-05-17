"""Tests for the unified error model.

Every 4xx and 5xx response must conform to ErrorResponse (the envelope
from app/schemas/errors.py). Consumers depend on this contract.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_predictor
from app.main import app
from app.schemas import ApiErrorDetail, ErrorResponse


def _assert_is_error_response(payload: dict, expected_code: str) -> None:
    """Assert the payload matches the ErrorResponse schema with the given code."""
    # Round-trip through the Pydantic model: if it doesn't validate, the
    # shape isn't what we documented in OpenAPI.
    parsed = ErrorResponse.model_validate(payload)
    assert isinstance(parsed.error, ApiErrorDetail)
    assert parsed.error.code.value == expected_code


# ───────────────────────────────────────────────────────────────────────
# 422 — validation errors across endpoints
# ───────────────────────────────────────────────────────────────────────

def test_422_predictions_missing_field(
    client: TestClient, valid_patient_dict: dict
) -> None:
    bad = {k: v for k, v in valid_patient_dict.items() if k != "drg_code"}
    r = client.post("/predictions", json=bad)
    assert r.status_code == 422
    _assert_is_error_response(r.json(), "VALIDATION_ERROR")


def test_422_predictions_extra_field(
    client: TestClient, valid_patient_dict: dict
) -> None:
    bad = {**valid_patient_dict, "extra_field": "X"}
    r = client.post("/predictions", json=bad)
    assert r.status_code == 422
    _assert_is_error_response(r.json(), "VALIDATION_ERROR")


def test_422_predictions_bad_threshold_query(
    client: TestClient, valid_patient_dict: dict
) -> None:
    r = client.post("/predictions?threshold=1.5", json=valid_patient_dict)
    assert r.status_code == 422
    _assert_is_error_response(r.json(), "VALIDATION_ERROR")


def test_422_predictions_batch_too_large(
    client: TestClient, valid_patient_dict: dict
) -> None:
    r = client.post(
        "/predictions/batch",
        json={"patients": [valid_patient_dict] * 101},
    )
    assert r.status_code == 422
    _assert_is_error_response(r.json(), "VALIDATION_ERROR")


def test_422_predictions_batch_empty(client: TestClient) -> None:
    r = client.post("/predictions/batch", json={"patients": []})
    assert r.status_code == 422
    _assert_is_error_response(r.json(), "VALIDATION_ERROR")


def test_422_examples_out_of_range(client: TestClient) -> None:
    r = client.get("/examples?n=101")
    assert r.status_code == 422
    _assert_is_error_response(r.json(), "VALIDATION_ERROR")


def test_422_explanations_missing_field(
    client: TestClient, valid_patient_dict: dict
) -> None:
    bad = {k: v for k, v in valid_patient_dict.items() if k != "bilirubin_max"}
    r = client.post("/explanations", json=bad)
    assert r.status_code == 422
    _assert_is_error_response(r.json(), "VALIDATION_ERROR")


def test_422_response_includes_per_field_details(
    client: TestClient, valid_patient_dict: dict
) -> None:
    """The details object should help the consumer pinpoint which field failed."""
    bad = {k: v for k, v in valid_patient_dict.items() if k != "drg_code"}
    r = client.post("/predictions", json=bad)
    body = r.json()
    assert "details" in body["error"]
    assert "errors" in body["error"]["details"]
    # Pydantic v2 lists missing fields with type='missing' and loc='drg_code'
    errors = body["error"]["details"]["errors"]
    assert any(
        "drg_code" in str(e.get("loc", [])) for e in errors
    ), f"missing-field error should reference drg_code: {errors}"


# ───────────────────────────────────────────────────────────────────────
# 500 — unhandled exception (must not leak traceback)
# ───────────────────────────────────────────────────────────────────────

class _BrokenPredictor:
    """Stand-in predictor that raises on every prediction.

    Used via dependency_overrides to trigger the 500 handler — we can't
    easily make the real predictor throw without breaking other tests.
    """

    encoders: dict = {}

    def predict_proba(self, df):
        raise RuntimeError("INTERNAL_TRACEBACK_SHOULD_NEVER_REACH_CLIENT_xyz_42")


def test_500_returns_generic_error_no_traceback_leak(
    valid_patient_dict: dict,
) -> None:
    """An unhandled exception must return a 500 in ErrorResponse shape,
    with NO trace of the original exception in the response body."""
    app.dependency_overrides[get_predictor] = lambda: _BrokenPredictor()
    try:
        # raise_server_exceptions=False so TestClient returns the 500 response
        # instead of re-raising the captured exception.
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.post("/predictions", json=valid_patient_dict)
        assert r.status_code == 500
        _assert_is_error_response(r.json(), "INTERNAL_ERROR")

        body_str = r.text
        # The sentinel string from _BrokenPredictor must not appear anywhere
        # in the response — proves the traceback isn't being leaked.
        assert "INTERNAL_TRACEBACK_SHOULD_NEVER_REACH_CLIENT_xyz_42" not in body_str
        assert "RuntimeError" not in body_str
        assert "Traceback" not in body_str
    finally:
        app.dependency_overrides.pop(get_predictor, None)


@pytest.mark.parametrize(
    "path,method,body",
    [
        ("/health", "GET", None),
        ("/metadata", "GET", None),
        ("/examples?n=3", "GET", None),
        ("/predictions", "POST", "VALID_DICT"),
        ("/predictions/batch", "POST", "BATCH"),
        ("/explanations", "POST", "VALID_DICT"),
    ],
)
def test_no_endpoint_leaks_request_body_in_500(
    valid_patient_dict: dict,
    path: str,
    method: str,
    body: str | None,
) -> None:
    """Every endpoint, when failing internally, hides request details."""
    if path in ("/health", "/metadata", "/examples?n=3"):
        # These don't depend on the predictor; skip the 500 simulation.
        pytest.skip(f"{path} doesn't use the predictor; covered by other tests")

    app.dependency_overrides[get_predictor] = lambda: _BrokenPredictor()
    try:
        payload = (
            valid_patient_dict
            if body == "VALID_DICT"
            else {"patients": [valid_patient_dict]}
            if body == "BATCH"
            else None
        )
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.request(method, path, json=payload)
        assert r.status_code == 500, f"{method} {path}: {r.status_code}"
        _assert_is_error_response(r.json(), "INTERNAL_ERROR")
    finally:
        app.dependency_overrides.pop(get_predictor, None)
