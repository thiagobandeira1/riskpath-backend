"""FastAPI app factory for the readmission prediction backend.

Builds the FastAPI app, registers CORS for local-dev frontends, wires
routers, and provides a lifespan placeholder. Task 3 extends `lifespan`
to pre-warm the `ReadmissionPredictor` so the first request doesn't pay
the ~3 s parquet load.

Run:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response

from app.error_handlers import register_exception_handlers
from app.routers import examples, health, metadata, predictions

# Permissive dev CORS, per spec.md Decisions Log #1. Tightened once the
# frontend stack is finalized. Covers the standard dev ports for the
# three frontends we might pick (React-Vite, Next/CRA, Streamlit).
_DEV_ORIGINS: list[str] = [
    "http://localhost:3000",   # Next.js / CRA
    "http://localhost:5173",   # Vite default
    "http://localhost:8080",   # @lovable.dev/vite-tanstack-config default (what RiskPath uses)
    "http://localhost:8501",   # Streamlit
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:8501",
]

# Default allowed-origin regex. Single regex covers every host shape we ship
# previews on, so we don't have to compose multiple regexes at startup (the
# previous compose-via-string-slicing approach silently dropped the leading 'h'
# from "https://" — every Lovable origin was being matched against "ttps://..."
# and CORS preflights failed). Override with ALLOWED_ORIGIN_REGEX env var if
# you tighten this for a specific deploy.
#
# Matches:
#   https://<uuid>.claudeusercontent.com           (Claude Design previews)
#   https://riskpath-clinician-companion.lovable.app  (Lovable hosted previews)
#   https://<name>.workers.dev / *.pages.dev       (Cloudflare Workers/Pages)
#   https://*.riskpath.app                         (future custom domain)
_DEFAULT_ALLOWED_ORIGIN_REGEX = (
    r"https://("
    r"[\w-]+\.claudeusercontent\.com|"
    r"([\w-]+\.)?(lovable\.app|workers\.dev|pages\.dev|riskpath\.app)"
    r")$"
)


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


def _resolve_cors(app: FastAPI) -> None:
    """Configure CORS from env vars (prod) with fallback to dev allowlist.

    ALLOWED_ORIGINS env var (comma-separated) is the prod allowlist. When set,
    it REPLACES the dev allowlist. The regex still allows Lovable/Workers
    preview URLs unless ALLOWED_ORIGIN_REGEX is also explicitly set.
    """
    env_origins = os.environ.get("ALLOWED_ORIGINS", "").strip()
    if env_origins:
        origins = [o.strip() for o in env_origins.split(",") if o.strip()]
    else:
        origins = _DEV_ORIGINS

    env_regex = os.environ.get("ALLOWED_ORIGIN_REGEX", _DEFAULT_ALLOWED_ORIGIN_REGEX)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=env_regex,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )


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
    _resolve_cors(app)

    # DUA mitigation: prevent search-engine indexing of every response.
    # Combined with /robots.txt below, this keeps the backend out of search
    # results so the MIMIC-derived /examples payload isn't crawler-surfaced.
    @app.middleware("http")
    async def add_no_index_header(request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        return response

    @app.get("/robots.txt", include_in_schema=False)
    def robots_txt() -> PlainTextResponse:
        """Forbid all crawling — DUA mitigation, not a security control."""
        return PlainTextResponse("User-agent: *\nDisallow: /\n")

    app.include_router(health.router)
    app.include_router(metadata.router)
    app.include_router(predictions.router)
    app.include_router(examples.router)

    register_exception_handlers(app)
    return app


app = create_app()
