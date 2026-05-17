"""GET /metadata — everything a frontend needs to build the input form.

Returns the 50 V7 features in canonical order, each with its type and
either (a) the list of training-time categorical levels or (b) the
numeric distribution stats from `model/missing_data_profile_v7.csv`.
Plus deployed-model summary + the default decision threshold.

The CSV + JSON metadata files are loaded once via lru_cache.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import get_predictor
from app.schemas import FeatureMetadata, MetadataResponse, ModelMetadata
from src.inference import ReadmissionPredictor
from src.schema import CATEGORICAL_COLS, FEATURE_COLS

router = APIRouter(tags=["metadata"])

_MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
_MISSING_DATA_CSV = _MODEL_DIR / "missing_data_profile_v7.csv"
_SUMMARY_JSON = _MODEL_DIR / "v7_summary.json"

_CATEGORICAL: set[str] = set(CATEGORICAL_COLS)

# Deployed seed-0 test AUROC. Documented in README + spec.md; the source of
# truth is verify_model.py, which recomputes it from the test split on each run.
_DEPLOYED_TEST_AUROC = 0.7929


@lru_cache(maxsize=1)
def _load_missing_data_profile() -> dict[str, dict[str, float]]:
    """Per-feature numeric stats loaded from missing_data_profile_v7.csv."""
    profile: dict[str, dict[str, float]] = {}
    with _MISSING_DATA_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            profile[row["feature"]] = {
                "min": float(row["min"]),
                "median": float(row["median"]),
                "max": float(row["max"]),
                "pct_nan": float(row["pct_nan"]),
            }
    return profile


@lru_cache(maxsize=1)
def _load_summary() -> dict:
    """Published 10-seed AUROC summary loaded from v7_summary.json."""
    return json.loads(_SUMMARY_JSON.read_text())


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    summary="Feature schema + model info",
    description=(
        "Returns the 50 V7 features (with categorical levels or numeric "
        "stats), the deployed model summary, and the default decision "
        "threshold. All a form-rendering frontend needs to build the "
        "input UI without inspecting src/ or model/."
    ),
)
def metadata(
    predictor: Annotated[ReadmissionPredictor, Depends(get_predictor)],
) -> MetadataResponse:
    profile = _load_missing_data_profile()
    summary = _load_summary()

    features: list[FeatureMetadata] = []
    for name in FEATURE_COLS:
        if name in _CATEGORICAL:
            encoder = predictor.encoders[name]
            features.append(
                FeatureMetadata(
                    name=name,
                    type="categorical",
                    levels=[str(c) for c in encoder.classes_],
                )
            )
        else:
            stats = profile.get(name)
            features.append(
                FeatureMetadata(
                    name=name,
                    type="numeric",
                    min=stats["min"] if stats else None,
                    median=stats["median"] if stats else None,
                    max=stats["max"] if stats else None,
                    pct_nan=stats["pct_nan"] if stats else None,
                )
            )

    xgb = summary["models"]["xgboost"]
    model_info = ModelMetadata(
        name="xgboost-v7-seed0",
        seed=0,
        n_features=len(FEATURE_COLS),
        published_test_auroc=float(xgb["test_auroc"]),
        deployed_test_auroc=_DEPLOYED_TEST_AUROC,
    )

    return MetadataResponse(
        features=features,
        model_info=model_info,
        default_threshold=0.5,
    )
