"""Response schemas for /predictions, /predictions/batch, /explanations,
/metadata, and /examples.

All response models have `description=` on every field — these descriptions
power Swagger UI at /docs and let any OpenAPI-driven consumer generate a
usable client without inspecting the source.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.features import PatientFeatures


# ───────────────────────────────────────────────────────────────────────
# Prediction
# ───────────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """30-day readmission probability + binary prediction for one patient."""

    model_config = ConfigDict(extra="forbid")

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="P(30-day readmission). XGBoost sigmoid output, [0, 1].",
    )
    prediction: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 if probability >= threshold, else 0.",
    )
    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Decision threshold actually used for this prediction.",
    )
    model_name: str = Field(
        default="xgboost-v7-seed0",
        description="Deployed model identifier.",
    )
    fallback_warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Per-feature warnings (e.g., 'drg_code value not seen in training; "
            "used most-common encoder fallback'). Empty for fully in-distribution input."
        ),
    )


class BatchPredictionResponse(BaseModel):
    """Batched predictions, one per input patient, in input order."""

    model_config = ConfigDict(extra="forbid")

    predictions: list[PredictionResponse] = Field(
        ...,
        description="One PredictionResponse per input patient, preserving input order.",
    )
    model_name: str = Field(
        default="xgboost-v7-seed0",
        description="Deployed model identifier.",
    )


# ───────────────────────────────────────────────────────────────────────
# Explanation
# ───────────────────────────────────────────────────────────────────────

class ExplanationResponse(BaseModel):
    """SHAP-based explanation for a single patient prediction."""

    model_config = ConfigDict(extra="forbid")

    shap_values: list[float] = Field(
        ...,
        description=(
            "Per-feature SHAP contributions (log-odds units) toward class 1. "
            "Length 50, in feature_names order. Sum + base_value ≈ model logit."
        ),
    )
    base_value: float = Field(
        ...,
        description="Model expected value (log-odds). The 'starting point' before features apply.",
    )
    feature_names: list[str] = Field(
        ...,
        description="Names of the 50 features, in canonical training order.",
    )
    feature_values_transformed: list[float] = Field(
        ...,
        description=(
            "The post-encoding values that went into the model (categoricals "
            "as integers). Same length and order as shap_values. Useful for "
            "rendering a waterfall chart with hover-over values."
        ),
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="P(30-day readmission) for the explained patient.",
    )
    model_name: str = Field(
        default="xgboost-v7-seed0",
        description="Deployed model identifier.",
    )
    fallback_warnings: list[str] = Field(
        default_factory=list,
        description="Same semantics as PredictionResponse.fallback_warnings.",
    )


# ───────────────────────────────────────────────────────────────────────
# Metadata
# ───────────────────────────────────────────────────────────────────────

class FeatureMetadata(BaseModel):
    """Per-feature info — enough for a frontend to build the input form."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Feature column name.")
    type: Literal["numeric", "categorical"] = Field(
        ...,
        description="Whether this feature is numeric or categorical.",
    )
    levels: list[str] | None = Field(
        default=None,
        description=(
            "For categorical features: full list of values seen during training "
            "(from LabelEncoder.classes_). None for numeric features."
        ),
    )
    min: float | None = Field(
        default=None,
        description="For numeric features: min observed value in training set.",
    )
    median: float | None = Field(
        default=None,
        description="For numeric features: median observed value.",
    )
    max: float | None = Field(
        default=None,
        description="For numeric features: max observed value.",
    )
    pct_nan: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="For numeric features: percentage of training rows where this is NaN.",
    )


class ModelMetadata(BaseModel):
    """Deployed model summary."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Model identifier.")
    seed: int = Field(..., description="Random seed used at training time.")
    n_features: int = Field(..., description="Number of input features expected.")
    published_test_auroc: float = Field(
        ...,
        description="10-seed averaged test AUROC from the publication (0.7935).",
    )
    deployed_test_auroc: float = Field(
        ...,
        description="Single-seed (seed-0) test AUROC for the shipped pickle (0.7929).",
    )


class MetadataResponse(BaseModel):
    """Everything a frontend needs to build the input form + display model info."""

    model_config = ConfigDict(extra="forbid")

    features: list[FeatureMetadata] = Field(
        ...,
        description="The 50 V7 features, in canonical training order.",
    )
    model_info: ModelMetadata = Field(
        ...,
        description="Deployed-model summary (named model_info to avoid Pydantic v2's `model_` shadow).",
    )
    default_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default decision threshold used when no `?threshold=` query param is given.",
    )


# ───────────────────────────────────────────────────────────────────────
# Examples
# ───────────────────────────────────────────────────────────────────────

class ExamplesResponse(BaseModel):
    """Pre-loaded anonymized patient rows for one-click demo loading.

    Per spec.md, the 5 ID_COLS (subject_id, hadm_id, admittime_dt,
    dischtime_dt, insurance) are stripped before serialization — every
    row matches the PatientFeatures shape exactly.
    """

    model_config = ConfigDict(extra="forbid")

    examples: list[PatientFeatures] = Field(
        ...,
        description=(
            "List of anonymized patient feature rows, each ready to POST to "
            "/predictions or /explanations without modification."
        ),
    )
    n: int = Field(..., ge=1, description="Number of examples returned.")
