"""Pydantic request/response/error schemas for the FastAPI backend."""

from app.schemas.errors import APIError, ApiErrorDetail, ErrorResponse
from app.schemas.features import PatientFeatures
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
    "BatchPredictionResponse",
    "ErrorResponse",
    "ExamplesResponse",
    "ExplanationResponse",
    "FeatureMetadata",
    "MetadataResponse",
    "ModelMetadata",
    "PatientFeatures",
    "PredictionResponse",
]
