"""Pydantic request/response/error schemas for the FastAPI backend."""

from app.schemas.errors import APIError, ApiErrorDetail, ErrorResponse
from app.schemas.features import MAX_BATCH_SIZE, BatchPredictionRequest, PatientFeatures
from app.schemas.responses import (
    BatchPredictionResponse,
    ExamplesResponse,
    ExplanationResponse,
    FeatureMetadata,
    MetadataResponse,
    ModelMetadata,
    PredictionResponse,
)

__all__ = [
    "APIError",
    "ApiErrorDetail",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "ErrorResponse",
    "ExamplesResponse",
    "ExplanationResponse",
    "FeatureMetadata",
    "MAX_BATCH_SIZE",
    "MetadataResponse",
    "ModelMetadata",
    "PatientFeatures",
    "PredictionResponse",
]
