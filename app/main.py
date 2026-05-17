"""FastAPI app factory for the readmission prediction backend.

Builds the FastAPI app, registers CORS for local-dev frontends, wires
routers, and provides a lifespan placeholder. Task 3 extends `lifespan`
to pre-warm the `ReadmissionPredictor` so the first request doesn't pay
the ~3 s parquet load.

Run:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import health, metadata, predictions
from app.schemas import APIError, ApiErrorDetail, ErrorResponse

# Permissive dev CORS, per spec.md Decisions Log #1. Tightened Saturday morning
# once the frontend stack is picked. Covers the standard dev ports for the
# three frontends we might pick (React-Vite, Next/CRA, Streamlit).
_DEV_ORIGINS: list[str] = [
    "http://localhost:3000",   # Next.js / CRA
    "http://localhost:5173",   # Vite
    "http://localhost:8501",   # Streamlit
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8501",
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Pre-warm predictor + SHAP explainer so the first request is fast."""
    import shap

    from app.dependencies import get_predictor

    predictor = get_predictor()
    # The TreeExplainer is otherwise lazy-built on first /explanations call
    # (per src/inference.py:68). Pre-building costs ~50 ms but keeps first-
    # request latency under the 2 s SHAP SLO.
    predictor._explainer = shap.TreeExplainer(predictor.model)
    yield


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    app = FastAPI(
        title="Readmission Prediction Backend",
        description=(
            "FastAPI backend wrapping the V7 seed-0 XGBoost 30-day Medicare "
            "readmission prediction model. See spec.md and plan.md at the "
            "project root for design context."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_DEV_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(metadata.router)
    app.include_router(predictions.router)

    # Minimal handler so all 422s match the documented ErrorResponse shape
    # (needed by the alignment tests in tests/test_feature_alignment.py).
    # Task 11 will move this to a dedicated module + add a 500 handler.
    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        body = ErrorResponse(
            error=ApiErrorDetail(
                code=APIError.VALIDATION_ERROR,
                message="Request validation failed",
                details={"errors": jsonable_encoder(exc.errors())},
            ),
        )
        return JSONResponse(status_code=422, content=jsonable_encoder(body))

    return app


app = create_app()
