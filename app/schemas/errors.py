"""Unified error response shape.

Every 4xx and 5xx response from the API returns this exact structure, per
spec.md §API and Interface Design — Consistent Error Semantics. Consumers
can rely on `response.json()["error"]["code"]` for programmatic error handling.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class APIError(str, Enum):
    """Machine-readable error codes returned in every error response."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ApiErrorDetail(BaseModel):
    """Inner error payload."""

    model_config = ConfigDict(extra="forbid")

    code: APIError = Field(
        ...,
        description="Machine-readable error code. Stable across API versions.",
    )
    message: str = Field(
        ...,
        description="Human-readable error description.",
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured detail (e.g., per-field validation errors).",
    )


class ErrorResponse(BaseModel):
    """Top-level error response envelope. Always returned for 4xx/5xx."""

    model_config = ConfigDict(extra="forbid")

    error: ApiErrorDetail = Field(..., description="Error payload.")
