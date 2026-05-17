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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health

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
    """App lifespan. Task 3 will pre-warm the predictor here."""
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
    return app


app = create_app()
