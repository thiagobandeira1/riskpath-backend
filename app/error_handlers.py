"""Centralized exception handlers.

Guarantees every 4xx/5xx response from the API matches the documented
ErrorResponse shape from app/schemas/errors.py — no matter where the
failure originates. Consumers can rely on `response.json()["error"]["code"]`
existing on every non-2xx response.

Two handlers:
    validation_exception_handler   422 — Pydantic validation failure
    unhandled_exception_handler    500 — anything else (logged in full,
                                          never leaked in the response body)
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import APIError, ApiErrorDetail, ErrorResponse

# Reusable `responses=` dict for endpoint decorators. Documents the
# ErrorResponse shape for 422 + 500 in OpenAPI so consumers see the error
# contract alongside the success contract.
ERROR_RESPONSES: dict = {
    422: {"model": ErrorResponse, "description": "Request validation failed."},
    500: {"model": ErrorResponse, "description": "Internal server error."},
}

_logger = logging.getLogger("uvicorn")


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Convert Pydantic RequestValidationError to a 422 in ErrorResponse shape."""
    body = ErrorResponse(
        error=ApiErrorDetail(
            code=APIError.VALIDATION_ERROR,
            message="Request validation failed",
            details={"errors": jsonable_encoder(exc.errors())},
        ),
    )
    return JSONResponse(status_code=422, content=jsonable_encoder(body))


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """500 handler — logs the full exception, returns a generic body.

    The exception type, message, and traceback NEVER appear in the response
    so that internal details can't leak to a malicious caller.
    """
    _logger.exception(
        "Unhandled exception in %s %s", request.method, request.url.path
    )
    body = ErrorResponse(
        error=ApiErrorDetail(
            code=APIError.INTERNAL_ERROR,
            message="Internal server error",
        ),
    )
    return JSONResponse(status_code=500, content=jsonable_encoder(body))


def register_exception_handlers(app: FastAPI) -> None:
    """Wire both handlers onto a FastAPI app."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
