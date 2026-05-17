"""Liveness endpoint.

Intentionally cheap — does NOT pre-warm the predictor or touch the parquet.
Returns 200 the moment uvicorn finishes binding the port.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.error_handlers import ERROR_RESPONSES

router = APIRouter(tags=["health"])

_DEPLOYED_MODEL_ID = "xgboost-v7-seed0"


class HealthResponse(BaseModel):
    """Service liveness + deployed model identifier."""

    status: str = Field(..., description='"ok" if the service is up.')
    model: str = Field(..., description="Deployed model identifier.")


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={500: ERROR_RESPONSES[500]},
    summary="Liveness probe",
    description="Returns 200 with the deployed model identifier. No model loading, no DB.",
)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model=_DEPLOYED_MODEL_ID)
