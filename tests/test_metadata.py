"""Tests for GET /metadata."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.schema import CATEGORICAL_COLS, FEATURE_COLS


def test_metadata_returns_200(client: TestClient) -> None:
    r = client.get("/metadata")
    assert r.status_code == 200


def test_metadata_lists_all_50_features_in_canonical_order(client: TestClient) -> None:
    r = client.get("/metadata")
    body = r.json()
    names = [f["name"] for f in body["features"]]
    assert names == FEATURE_COLS  # exact order matters


def test_metadata_categorical_features_have_levels(client: TestClient) -> None:
    r = client.get("/metadata")
    body = r.json()
    cats = {f["name"]: f for f in body["features"] if f["type"] == "categorical"}

    assert set(cats.keys()) == set(CATEGORICAL_COLS)
    for name, f in cats.items():
        assert f["levels"] is not None, f"categorical {name} missing levels"
        assert len(f["levels"]) > 0, f"categorical {name} has empty levels"
        # Categoricals should not carry numeric stats
        assert f["min"] is None
        assert f["median"] is None
        assert f["max"] is None
        assert f["pct_nan"] is None


def test_metadata_numeric_features_have_distribution_stats(client: TestClient) -> None:
    r = client.get("/metadata")
    body = r.json()
    nums = [f for f in body["features"] if f["type"] == "numeric"]

    assert len(nums) == 50 - len(CATEGORICAL_COLS)  # 46
    for f in nums:
        assert f["levels"] is None, f"numeric {f['name']} should not have levels"
        # CSV covers all 46 numeric features — every numeric field should
        # come back with min/median/max/pct_nan populated.
        assert f["min"] is not None, f"numeric {f['name']} missing min"
        assert f["median"] is not None, f"numeric {f['name']} missing median"
        assert f["max"] is not None, f"numeric {f['name']} missing max"
        assert f["pct_nan"] is not None, f"numeric {f['name']} missing pct_nan"
        assert f["min"] <= f["median"] <= f["max"], f"{f['name']}: min/median/max out of order"
        assert 0.0 <= f["pct_nan"] <= 100.0


def test_metadata_model_info(client: TestClient) -> None:
    r = client.get("/metadata")
    body = r.json()
    m = body["model_info"]
    assert m["name"] == "xgboost-v7-seed0"
    assert m["seed"] == 0
    assert m["n_features"] == 50
    # AUROC sanity bounds — generous on purpose; the exact values are
    # verified by verify_model.py, not by this test.
    assert 0.78 < m["published_test_auroc"] < 0.80
    assert 0.78 < m["deployed_test_auroc"] < 0.80


def test_metadata_default_threshold_is_half(client: TestClient) -> None:
    r = client.get("/metadata")
    assert r.json()["default_threshold"] == 0.5


def test_metadata_no_phi_keys_in_response(client: TestClient) -> None:
    """Sanity guard: metadata should never expose patient identifiers."""
    r = client.get("/metadata")
    body_str = r.text.lower()
    for risky in ("subject_id", "hadm_id", "admittime_dt", "dischtime_dt", "insurance"):
        assert risky not in body_str, f"{risky!r} leaked into /metadata response"
