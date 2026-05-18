# syntax=docker/dockerfile:1.6
#
# Production container for the RiskPath readmission-prediction backend.
#
# Design choices:
#   * python:3.11-slim (Debian bookworm slim, ~50 MB base, matches local venv).
#   * libgomp1 is the only system runtime dep XGBoost needs; gcc + python-dev
#     are needed in the build stage to compile any wheels that lack manylinux2014
#     binaries for arm64.
#   * Multi-stage so the final image doesn't carry the build toolchain.
#   * No parquet copied in. `model/deploy_encoders.pkl` and
#     `model/deploy_examples_pool.pkl` (pre-baked locally by
#     `scripts/freeze_deploy_artifacts.py`) provide everything the runtime needs.
#   * uvicorn binds to $PORT — Railway, Fly.io, Render and Cloud Run all set this.

# ──────────────────────────── build stage ────────────────────────────
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential covers gcc/g++/make for any source wheels (rare on amd64,
# common on arm64 — Railway runs amd64 today, kept as defensive armor).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install --upgrade pip \
    && pip install --prefix=/install -r requirements.txt

# ───────────────────────── runtime stage ─────────────────────────────
FROM python:3.11-slim AS runtime

# libgomp1: XGBoost runtime requirement (OpenMP).
# tini: PID-1 / signal-handling so uvicorn exits cleanly on SIGTERM.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copy installed Python packages from the builder stage.
COPY --from=builder /install /usr/local

# Application code — only the runtime-needed paths.
# `data/` is intentionally NOT copied (parquet would breach the PhysioNet DUA
# and bloat the image by 25 MB; the pre-baked pickles in `model/` cover all
# runtime needs).
COPY app/   ./app/
COPY src/   ./src/
COPY model/ ./model/

# Non-root user for principle-of-least-privilege.
RUN useradd --create-home --uid 1001 riskpath \
    && chown -R riskpath:riskpath /app
USER riskpath

# Railway / Fly / Cloud Run / Render all set $PORT. Default 8000 for local
# `docker run` without -e PORT=...
ENV PORT=8000
EXPOSE 8000

# Health checks hit /health (already implemented; no DB or cache deps).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
        sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=3).status == 200 else 1)"

ENTRYPOINT ["/usr/bin/tini", "--"]
# Single worker on purpose: the predictor + SHAP TreeExplainer are pre-warmed
# in `app.main:lifespan` and cached at module scope; multi-worker would
# duplicate ~400 MB of RAM per worker. Scale horizontally via the platform if
# load demands it.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
