# Spec: Readmission Prediction Backend (FastAPI)

A FastAPI HTTP backend that wraps the deployed V7 seed-0 XGBoost model
(`src/inference.py:ReadmissionPredictor`) and exposes prediction + SHAP
explanation endpoints. Designed to be consumed by a single browser-based
form-driven frontend (stack TBD — Streamlit, React, Vue, or generated from
OpenAPI), demoed live to a healthcare executive on Saturday.

**Author:** Thiago Bandeira  •  **Drafted:** 2026-05-17  •  **Status:** Draft, awaiting approval

---

## Assumptions in this draft

Per the spec-driven-development skill, surfacing inferred assumptions up front
so they can be corrected *before* code:

1. **Stack is FastAPI** (per "Then proceed to drafting spec.md" framing) — not Streamlit.
2. **Single browser-based form-driven consumer, stack TBD.** The frontend choice (Streamlit, React from Claude Design, hand-written, etc.) is deferred and is not on the backend's critical path. Backend shape — strong typing, OpenAPI quality, a `/metadata` endpoint, snake_case JSON — is the same regardless of which frontend is picked. CORS is stubbed permissively for the prototype and tightened Saturday morning once the frontend exists.
3. **The demo is local-only Saturday** — backend running on `127.0.0.1:8000` from your laptop, frontend running locally or via Claude Design's hosted preview. No deployment, no public URL, no auth, no TLS.
4. **No persistence layer.** No database, no request logging, no audit trail. Each request is stateless: features in → prediction + SHAP out.
5. **No real PHI in transit.** Inputs are either (a) synthetic feature vectors typed into the UI, or (b) rows drawn from the local V7 parquet by row index. Either way the FastAPI process never sees patient identifiers — only the 50 V7 feature columns.
6. **Encoders fit at startup, once.** `build_encoders()` reads the V7 parquet at app boot (~3s, 244,576 rows × 4 cat cols). The parquet path stays the same as in `src/preprocess.py`. The backend is unusable if the parquet is missing — that's intentional, surfaces the DUA dependency.
7. **No batch sizes > 100.** SHAP on the full test set (49,191 rows) is slow; we cap at 100 rows per request to keep p95 latency under 2 s.
8. **Python 3.12.10, the existing venv.** No second interpreter, no Docker.

> Correct any of these now — they're load-bearing. The "Open Questions" section at the bottom flags decisions I'm leaving for you.

---

## Objective

Ship a working local HTTP API by Friday EOD that:

- Loads the V7 seed-0 XGBoost model once at startup
- Accepts patient feature vectors and returns a 30-day readmission probability + binary prediction
- Returns SHAP explanations (per-feature contributions, base value, feature names) for the same input
- Exposes a `/metadata` endpoint that gives any form-rendering frontend what it needs to build the input form (feature names, types, categorical levels, summary stats from `missing_data_profile_v7.csv`)
- Exposes a `GET /examples?n=5` endpoint that returns anonymized rows drawn from the local V7 parquet — the 50 feature columns only, with all 5 `ID_COLS` (`subject_id`, `hadm_id`, `admittime_dt`, `dischtime_dt`, `insurance`) stripped before serialization. Lets the demo executive click "Load example" instead of typing 50 values.
- Is documented via OpenAPI at `/docs` (Swagger UI) and `/openapi.json` (machine-readable for code-gen consumers)
- Is testable end-to-end in under 60 s via `pytest`

Saturday's demo flow (drives every scope call below):

> A healthcare executive sees a single-patient form, enters or selects a patient, clicks Predict, sees a probability + a SHAP waterfall identifying the top 5 risk drivers, and asks "what if we changed *X*?" — at which point we tweak one feature, re-submit, and the probability moves visibly.

**Non-goals (explicit):** authentication, multi-tenancy, persistence, request logging, deployment, rate-limiting, model retraining, model versioning UI, A/B testing of thresholds.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| HTTP framework | **FastAPI ≥0.115** | Pydantic v2 native, auto OpenAPI for any consumer |
| ASGI server | **uvicorn[standard] ≥0.32** | FastAPI's reference server, `--reload` for dev |
| Schema/validation | **Pydantic 2.x** | Bundled with FastAPI; shapes the API contract |
| Test framework | **pytest** + **httpx** (via `TestClient`) | In-process API tests, no live server needed |
| Model + SHAP | unchanged — `src/inference.py:ReadmissionPredictor` | Already verified end-to-end (see `verify_model.py`) |
| Python | **3.12.10** (existing `./venv`) | Matches the pinned env |

To add to `requirements.txt`:

```
fastapi>=0.115,<0.120
uvicorn[standard]>=0.32,<1.0
pytest>=8.0
httpx>=0.27          # used by FastAPI TestClient
```

Pinned to compatible major bands rather than exact pins because these are new additions to the env, not part of the V17-validated training-time pins.

---

## Commands

All commands run from project root, with `./venv` activated:

```powershell
# --- Activate venv (every new shell) ---
.\venv\Scripts\Activate.ps1

# --- Dev server (auto-reload on file change) ---
uvicorn app.main:app --reload --port 8000

# --- Production-ish run (single worker, no reload) ---
uvicorn app.main:app --host 127.0.0.1 --port 8000

# --- Run tests (full suite, ~30 s target) ---
pytest -v

# --- Run only the non-negotiable feature-alignment tests ---
pytest tests/test_feature_alignment.py -v

# --- Re-run the model verification (sanity check before each demo) ---
python verify_model.py

# --- Check OpenAPI schema is generated correctly ---
python -c "from app.main import app; import json; print(json.dumps(app.openapi(), indent=2))" > openapi.json
```

Linting/formatting deferred — pinned tooling adds setup time that 5 days can't absorb. Style enforced by code review, not tooling.

---

## Project Structure

Extends the existing layout. New paths in **bold**, existing untouched.

```
Hospital Predictive Model App/
├── app/                                ← NEW: FastAPI application
│   ├── __init__.py
│   ├── main.py                         ← FastAPI app factory + CORS + lifespan
│   ├── dependencies.py                 ← Singleton ReadmissionPredictor injector
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── health.py                   ← GET /health
│   │   ├── metadata.py                 ← GET /metadata  (feature schema for UI)
│   │   ├── examples.py                 ← GET /examples?n=N  (anonymized demo rows)
│   │   └── predictions.py              ← POST /predictions, /predictions/batch, /explanations
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── features.py                 ← PatientFeatures Pydantic model (50 fields)
│   │   ├── responses.py                ← PredictionResponse, ExplanationResponse, etc.
│   │   └── errors.py                   ← ErrorResponse + APIError
│   └── serialization.py                ← numpy → JSON helpers (SHAP arrays, etc.)
│
├── tests/                              ← NEW: pytest suite
│   ├── __init__.py
│   ├── conftest.py                     ← TestClient + fixture patient rows
│   ├── test_feature_alignment.py       ← NON-NEGOTIABLE (see Testing Strategy)
│   ├── test_predictions.py
│   ├── test_explanations.py
│   ├── test_metadata.py
│   ├── test_examples.py                ← asserts ID_COLS never appear in responses
│   └── test_errors.py
│
├── spec.md                             ← NEW: this document
├── model/                              ← unchanged
├── src/                                ← unchanged — do NOT touch
├── data/                               ← unchanged, gitignored
├── notebooks/                          ← unchanged
├── verify_model.py                     ← unchanged
├── requirements.txt                    ← extended (4 new packages)
├── README.md                           ← appended with "Running the backend" section at the end
└── venv/                               ← gitignored
```

**Critical rule:** `src/` is the model contract — frozen for this build. The backend imports from `src/`; it does not modify it. If `src/` needs to change, that's a separate decision and a separate spec.

---

## Code Style

Match the conventions already in `src/inference.py`, `src/schema.py`, `src/preprocess.py`:

```python
"""Module-level docstring describing purpose and usage."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.dependencies import get_predictor
from src.inference import ReadmissionPredictor

router = APIRouter(prefix="/predictions", tags=["predictions"])


class PredictionResponse(BaseModel):
    """30-day readmission probability + binary prediction for one patient."""

    probability: float = Field(..., ge=0.0, le=1.0, description="P(readmit within 30 days)")
    prediction: int = Field(..., ge=0, le=1, description="1 if probability >= threshold, else 0")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Decision threshold used")
    model_name: str = Field(default="xgboost-v7-seed0")


@router.post("", response_model=PredictionResponse)
def predict_single(
    features: PatientFeatures,
    predictor: Annotated[ReadmissionPredictor, Depends(get_predictor)],
    threshold: float = 0.5,
) -> PredictionResponse:
    """Predict 30-day readmission probability for one patient."""
    df = features.to_dataframe()
    proba = float(predictor.predict_proba(df)[0])
    return PredictionResponse(
        probability=proba,
        prediction=int(proba >= threshold),
        threshold=threshold,
    )
```

Conventions:

- `from __future__ import annotations` at the top of every Python file
- Type hints on every public function/method, including return types
- `pathlib.Path` for file paths, never raw strings
- Module-private constants prefixed `_` (`_MODEL_DIR`, `_DEFAULT_THRESHOLD`)
- Keyword-only args after `*` for any function with > 2 args
- Snake_case for both Python identifiers AND JSON keys — Pydantic field names *are* the JSON keys; using snake_case for both keeps Python and JSON one-to-one, matches FastAPI defaults, and is fine for every plausible consumer (Streamlit, React-from-OpenAPI, plain curl)
- Pydantic `Field(...)` with `description=` and validation bounds on every field — form-rendering frontends and OpenAPI code-gen consumers both use these
- Lazy imports only for heavy/optional deps (matches the `import shap` pattern in `src/inference.py:66`)
- Specific exceptions over `Exception` — `ValueError` for bad inputs, `HTTPException` for HTTP-translatable failures
- One short docstring per public function/class; skip docstrings on test functions (test name *is* the docstring)
- No comments explaining *what* code does — only *why*, and only when non-obvious

---

## Testing Strategy

`pytest` + FastAPI's `TestClient` (synchronous, in-process — no live server). Test files live in `tests/`, mirror the router structure. Target: full suite under 60 s.

### Coverage by layer

| Concern | File | Approach |
|---|---|---|
| Feature alignment (see below) | `test_feature_alignment.py` | **NON-NEGOTIABLE** — see dedicated section |
| Endpoint contracts | `test_predictions.py`, `test_explanations.py`, `test_metadata.py` | Happy path + ≥2 edge cases per endpoint |
| Examples endpoint (PHI guard) | `test_examples.py` | `GET /examples?n=N` for N ∈ {1, 5, 100}; assert response is a list of length N; assert **none** of the 5 `ID_COLS` appear as keys in any returned object; assert each object has all 50 feature columns |
| Error responses | `test_errors.py` | Each 4xx code reachable and returns the documented `ErrorResponse` shape |
| Predictor singleton | `test_predictions.py` | Asserts `get_predictor()` returns the same instance across requests |

### Non-negotiable: feature-alignment test coverage

These tests *must* exist and pass before any commit that touches the prediction path. The failure mode they guard against is the one no human review will catch: the FastAPI route silently dropping, reordering, or wrong-typing the 50 features relative to `model/v7_feature_cols.json`, producing **plausible but wrong** probabilities.

Required tests in `tests/test_feature_alignment.py`:

1. **`test_column_order_matches_model`** — POST 5 patient rows where the JSON keys are sent in **shuffled order**. Assert the returned probabilities are identical (to 1e-9) to a control request where the same rows are sent in canonical order. *(Guards against dict-iteration-order assumptions producing column-shuffled input.)*

2. **`test_categorical_encoding_matches_training`** — For each of the 4 categorical columns (`drg_code`, `last_drg_dispo`, `discharge_location`, `primary_dx_chapter`), send a row where that column has a value present in the training set. Assert the predicted probability matches `ReadmissionPredictor` called directly on the same row. *(Guards against the FastAPI layer applying different LabelEncoders than the predictor uses.)*

3. **`test_missing_required_feature_returns_422`** — POST a request missing one of the 50 features. Assert HTTP 422 and `error.code == "VALIDATION_ERROR"`. *(Guards against silent default-filling.)*

4. **`test_extra_unknown_field_is_rejected`** — POST a request with all 50 features plus one unknown field (`"surgeon_name": "X"`). Assert HTTP 422. *(Pydantic `model_config = ConfigDict(extra="forbid")` — guards against typos in real field names being silently ignored.)*

5. **`test_unseen_categorical_value_handled`** — POST a row where `drg_code` is a string never seen in training. Assert the endpoint returns a valid probability (per `preprocess.transform`'s `most_common` fallback) AND the response includes a warning field flagging the fallback occurred. *(Surfaces the silent-fallback behavior to the consumer rather than hiding it.)*

6. **`test_full_test_set_auroc_matches_inference`** — Load 1,000 rows from the V7 test split, POST each via `/predictions`, collect probabilities, compute AUROC. Assert it matches `predictor.predict_proba()` called directly on those rows (to 1e-9). *(End-to-end correctness — guarantees the HTTP path is mathematically equivalent to the verified `ReadmissionPredictor`.)*

If any of these six tests is skipped, disabled, or marked xfail without explicit written justification in the spec, that's a stop-the-build event.

### Test data

Test rows come from the V7 parquet via `df.sample(n=20, random_state=42)` inside a session-scoped `conftest.py` fixture. **Never** hardcoded in test files (would commit MIMIC rows). Fixture loads the parquet once per test session, slices into ≤20 row samples per test.

### What's NOT tested (deliberately)

- Performance/load (single-user demo, not relevant)
- Concurrency (synchronous TestClient, single worker)
- Auth (none in this build)
- The model itself (already covered by `verify_model.py`)

---

## Boundaries

Compressed per request. PHI handling first because it's the only one that's legally binding.

**Never:**
- Log or print request bodies (could contain MIMIC-derived feature values). Configure uvicorn with `--log-level warning` and ensure no `print(request)` survives review.
- Persist requests or responses to disk in any form (no SQLite, no JSONL audit log, no temp files).
- Expose `subject_id`, `hadm_id`, `admittime_dt`, `dischtime_dt`, or `insurance` in any response — `ID_COLS` from `src/schema.py` must be filtered out of `/examples` responses before serialization. `test_examples.py` enforces this.
- Add the V7 parquet path, `data/`, or any patient row to git. `.gitignore` already enforces this; verify before each `git add`.
- Push to a remote — no GitHub remote exists yet, and won't until the build is stable and we've re-audited.
- Modify `src/` (frozen — the trained-model contract).
- Add features that aren't in this spec without amending the spec first.

**Ask first:**
- Adding a new dependency to `requirements.txt`
- Changing the request/response shape after the first endpoint is shipped (Hyrum's Law applies even with one consumer)
- Touching `model/` files
- Adding a remote git endpoint (private or otherwise)

**Always:**
- Run `pytest tests/test_feature_alignment.py` before committing any change to the prediction path
- Run `verify_model.py` after any change that imports from `src/`
- Activate `./venv` before any `python`/`pytest`/`uvicorn` command
- Commit each vertical slice separately (per `incremental-implementation` skill)

---

## Success Criteria

The backend is done when **all** of these are true:

- [ ] `uvicorn app.main:app` starts in < 10 s, exits cleanly on Ctrl+C, and serves `/docs` at `http://127.0.0.1:8000/docs`
- [ ] `pytest` runs to completion in < 60 s with 0 failures, 0 skips, ≥ 25 tests
- [ ] All 6 feature-alignment tests pass
- [ ] `POST /predictions` p95 latency < 200 ms (single patient, after warm-up)
- [ ] `POST /explanations` p95 latency < 2 s (single patient, includes SHAP)
- [ ] `GET /metadata` returns enough information for any form-rendering frontend to build the input form without inspecting `src/` or `model/`
- [ ] `GET /examples?n=N` returns N anonymized rows with all 5 `ID_COLS` stripped; verified by `test_examples.py`
- [ ] OpenAPI schema validates against the OpenAPI 3.1 spec (FastAPI gives this for free)
- [ ] README has a "Running the backend" section with the 3 commands needed for the demo
- [ ] No request body or feature value appears in any log line at any log level
- [ ] `git log` shows one vertical-slice commit per task (no "WIP" or "fix" commits)

---

## Decisions Log

All open questions from the initial draft were resolved 2026-05-17:

| # | Decision | Rationale |
|---|---|---|
| 1 | **Frontend stack deferred.** Backend is consumer-agnostic. CORS shipped with a permissive dev stub (`["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:*"]`), tightened Saturday morning when the stack is picked. | Backend isn't blocked by the frontend choice; CORS is a 2-line config change. |
| 2 | **Threshold default = 0.5, exposed as `?threshold=` query param.** | Industry-standard default + per-request tunability. Doesn't bake in a clinical operating point the model wasn't validated for. |
| 3 | **`GET /examples?n=N` endpoint included.** Returns N rows from the V7 parquet with all 5 `ID_COLS` stripped. Local-backend → local-browser flow, no network redistribution. | Demo executive clicks "Load example" instead of typing 50 fields. Saturday demo significantly smoother. |
| 4 | **General SHAP narrative for the demo** — any patient → probability → top 5 SHAP drivers → tweak a feature → watch probability move. | Canonical AI-explainability demo, no domain prep needed, works with any pre-loaded example. |
| 5 | **`spec.md` committed to git** as the second commit. Slice commits reference its sections. | Source of truth for downstream work; audit trail for the engineering process. |

---

*End of spec. Awaiting approval before moving to the implementation plan and task breakdown.*
