"""Prediction endpoints.

POST /predictions             single-patient probability + binary decision
POST /predictions/batch       (added in Task 8)
POST /explanations            (added in Task 9)

The handler converts PatientFeatures to a 1-row DataFrame in canonical
FEATURE_COLS order (via PatientFeatures.to_dataframe()), then delegates
to the shared ReadmissionPredictor. Unseen categorical values are
detected here and surfaced to the consumer via `fallback_warnings`.
"""

from __future__ import annotations

from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, Query

from app.dependencies import get_predictor
from app.schemas import PatientFeatures, PredictionResponse
from src.inference import ReadmissionPredictor
from src.schema import CATEGORICAL_COLS

router = APIRouter(tags=["predictions"])


def _detect_fallback_warnings(
    df: pd.DataFrame, predictor: ReadmissionPredictor
) -> list[str]:
    """Per-column warnings for categorical values not seen during training.

    The underlying preprocess.transform applies a most-common fallback
    silently; surfacing the warning to the consumer is the only way the
    UI can flag that the prediction relied on a degraded encoding.
    """
    warnings: list[str] = []
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        encoder = predictor.encoders[col]
        known = set(encoder.classes_)
        seen = set(df[col].astype(str))
        unknown = sorted(seen - known)
        if unknown:
            preview = unknown[:3]
            warnings.append(
                f"{col}: value(s) {preview} not seen in training; "
                f"used most-common fallback encoding"
            )
    return warnings


@router.post(
    "/predictions",
    response_model=PredictionResponse,
    summary="30-day readmission probability + binary prediction (single patient)",
    description=(
        "Returns the predicted probability of 30-day readmission and the "
        "binary decision at the requested threshold. Categorical values "
        "not seen during training are accepted but surfaced via "
        "`fallback_warnings` so the consumer can flag a degraded prediction."
    ),
)
def predict_single(
    features: PatientFeatures,
    predictor: Annotated[ReadmissionPredictor, Depends(get_predictor)],
    threshold: Annotated[
        float,
        Query(
            ge=0.0,
            le=1.0,
            description=(
                "Decision threshold for the binary prediction. "
                "Default 0.5; expose to clinicians as a tunable knob."
            ),
        ),
    ] = 0.5,
) -> PredictionResponse:
    df = features.to_dataframe()
    warnings = _detect_fallback_warnings(df, predictor)
    proba = float(predictor.predict_proba(df)[0])
    return PredictionResponse(
        probability=proba,
        prediction=int(proba >= threshold),
        threshold=threshold,
        fallback_warnings=warnings,
    )
