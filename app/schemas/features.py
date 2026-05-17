"""Input schema: PatientFeatures.

50 fields, generated programmatically from `src.schema.FEATURE_COLS` so the
model definition stays in lockstep with the trained model's feature list —
no risk of the schema drifting out of sync with the pickle.

Field types:
    - 4 categorical features (drg_code, last_drg_dispo, discharge_location,
      primary_dx_chapter) are typed `str`. Unknown values are NOT rejected
      at the Pydantic layer — `preprocess.transform()` applies a most-common
      fallback and the prediction endpoint surfaces this as a warning in
      the response (see test_feature_alignment.py: test_unseen_categorical).
    - 46 numeric features are typed `float | None`. NaN is a valid value
      in the source data, so the consumer may legitimately send `null` for
      missing measurements.

Strictness:
    - All 50 fields are required (no defaults). Missing a key returns 422.
    - extra="forbid" — unknown field names return 422 instead of being
      silently dropped. Guards against typos like {"drg_codes": ...} that
      would otherwise produce plausible-but-wrong predictions.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, create_model

from src.schema import CATEGORICAL_COLS, FEATURE_COLS

_CATEGORICAL = set(CATEGORICAL_COLS)


class _StrictBase(BaseModel):
    """Base for PatientFeatures — forbids extra keys."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    def to_dataframe(self) -> pd.DataFrame:
        """Return this row as a 1-row DataFrame ready for the predictor.

        Numeric columns are explicitly cast to float64 so None becomes NaN
        rather than leaving the column as object dtype — XGBoost rejects
        object-dtype numeric columns with `KeyError: 'object'`.
        Categorical columns stay as object/str; preprocess.transform applies
        the LabelEncoder.
        """
        row = self.model_dump()
        df = pd.DataFrame([{c: row[c] for c in FEATURE_COLS}], columns=FEATURE_COLS)
        numeric_cols = [c for c in FEATURE_COLS if c not in _CATEGORICAL]
        df[numeric_cols] = df[numeric_cols].astype("float64")
        return df


def _build_field_defs() -> dict[str, tuple[type, object]]:
    """Construct the (type, FieldInfo) mapping for create_model."""
    defs: dict[str, tuple[type, object]] = {}
    for name in FEATURE_COLS:
        if name in _CATEGORICAL:
            defs[name] = (
                str,
                Field(..., description=f"V7 categorical feature: {name}"),
            )
        else:
            # `float | None` because the source data legitimately has NaN for
            # many lab values and engineered features. Required — consumer
            # must explicitly send null, not omit the key.
            defs[name] = (
                float | None,
                Field(..., description=f"V7 numeric feature: {name} (null = missing)"),
            )
    return defs


PatientFeatures = create_model(
    "PatientFeatures",
    __base__=_StrictBase,
    **_build_field_defs(),
)
PatientFeatures.__doc__ = (
    "Single patient's 50 V7 feature values. See /metadata for per-feature "
    "type, value range, and (for categoricals) the list of known levels."
)


MAX_BATCH_SIZE = 100


class BatchPredictionRequest(BaseModel):
    """Request body for POST /predictions/batch (1-100 patients per call)."""

    model_config = ConfigDict(extra="forbid")

    patients: list[PatientFeatures] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=(
            f"List of patient feature objects (1-{MAX_BATCH_SIZE} per request). "
            "SHAP latency grows linearly; the cap protects the per-request "
            "latency target."
        ),
    )
