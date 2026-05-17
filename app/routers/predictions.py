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
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExplanationResponse,
    PatientFeatures,
    PredictionResponse,
)
from app.serialization import ndarray_to_list
from src.inference import ReadmissionPredictor
from src.schema import CATEGORICAL_COLS, FEATURE_COLS

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


@router.post(
    "/predictions/batch",
    response_model=BatchPredictionResponse,
    summary="30-day readmission predictions (1-100 patients per call)",
    description=(
        "Batched version of POST /predictions. The full batch is scored in "
        "a single XGBoost forward pass, so per-patient overhead is much "
        "lower than calling /predictions N times. Cap is 100 patients "
        "per request to keep per-request latency bounded."
    ),
)
def predict_batch(
    request: BatchPredictionRequest,
    predictor: Annotated[ReadmissionPredictor, Depends(get_predictor)],
    threshold: Annotated[
        float,
        Query(ge=0.0, le=1.0, description="Decision threshold applied to every patient."),
    ] = 0.5,
) -> BatchPredictionResponse:
    # Build one multi-row DataFrame so we do a single forward pass.
    df = pd.concat(
        [pf.to_dataframe() for pf in request.patients], ignore_index=True
    )
    df = df[FEATURE_COLS]  # belt-and-braces: enforce canonical order

    probas = predictor.predict_proba(df)
    predictions = [
        PredictionResponse(
            probability=float(p),
            prediction=int(p >= threshold),
            threshold=threshold,
            fallback_warnings=_detect_fallback_warnings(df.iloc[[i]], predictor),
        )
        for i, p in enumerate(probas)
    ]
    return BatchPredictionResponse(predictions=predictions)


@router.post(
    "/explanations",
    response_model=ExplanationResponse,
    summary="SHAP explanation for a single patient prediction",
    description=(
        "Returns per-feature SHAP contributions (in log-odds units) toward "
        "the positive class, the model expected value (base_value), the "
        "post-encoding feature values that went into the model, and the "
        "predicted probability. sum(shap_values) + base_value equals the "
        "model logit; sigmoid(.) of that equals the probability."
    ),
)
def explain_single(
    features: PatientFeatures,
    predictor: Annotated[ReadmissionPredictor, Depends(get_predictor)],
) -> ExplanationResponse:
    df = features.to_dataframe()
    warnings = _detect_fallback_warnings(df, predictor)

    shap_out = predictor.explain(df)
    # shap_values shape: (1, 50). Extract the row as a flat list.
    shap_row = ndarray_to_list(shap_out["shap_values"][0])
    # X_transformed: (1, 50) DataFrame in FEATURE_COLS order.
    x_row = ndarray_to_list(shap_out["X_transformed"].iloc[0].to_numpy())

    proba = float(predictor.predict_proba(df)[0])

    return ExplanationResponse(
        shap_values=shap_row,
        base_value=float(shap_out["base_value"]),
        feature_names=list(shap_out["feature_names"]),
        feature_values_transformed=x_row,
        probability=proba,
        fallback_warnings=warnings,
    )
