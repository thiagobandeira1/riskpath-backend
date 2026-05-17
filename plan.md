# Implementation Plan: FastAPI Backend

Maps `spec.md` to 14 ordered, vertically-sliced tasks across 3 phases.
Drafted 2026-05-17. Single developer, 5-day window, day job during the week.

---

## Overview

Build the 5-endpoint FastAPI backend specified in `spec.md`: `/health`,
`/metadata`, `/predictions`, `/predictions/batch`, `/explanations`, `/examples`.
Each implementation slice is one router file + one test file + (where needed)
one schema file. The model layer (`src/`) is frozen — never modified.

## Architecture Decisions

| Decision | Rationale |
|---|---|
| **Singleton predictor via FastAPI dependency injection** (one `ReadmissionPredictor` instance for the process lifetime, built at startup via `lifespan`) | `build_encoders()` reads the V7 parquet at boot (~3 s, 244k rows). Re-doing this per request would make latency unusable. |
| **`functools.lru_cache(maxsize=1)` on the predictor factory** | Idiomatic FastAPI singleton pattern; works with `Depends()`; testable via dependency override. |
| **Tests use FastAPI `TestClient` (synchronous, in-process)** | No live server needed; faster suite; no port conflicts. |
| **Test data fixtures load V7 parquet once per session** | Avoids re-reading 244k rows for every test; never hardcodes patient rows into test files (would leak via git). |
| **`/predictions/batch` is a separate endpoint, not a `?batch=true` flag** | Per `api-and-interface-design`: endpoints with different request shapes should be different endpoints. |
| **Pydantic `model_config = ConfigDict(extra="forbid")` on `PatientFeatures`** | Mistyped field names (e.g., `"drg_codes"` instead of `"drg_code"`) fail loud at the API boundary instead of silently dropping. |
| **SHAP numpy arrays serialized via a dedicated `app/serialization.py` module** | Pydantic v2 can't natively serialize `np.ndarray`; one shared helper avoids three subtly-different implementations across endpoints. |

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Feature column reorder produces plausible-but-wrong predictions | **Critical** — silent correctness failure | The 6 non-negotiable alignment tests in Task 7. Task 7 cannot be marked complete without all 6 green. |
| `/examples` accidentally leaks `ID_COLS` to the browser | DUA violation | `test_examples.py` asserts none of the 5 `ID_COLS` keys appear in any response object. Task 10 cannot be marked complete without this test green. |
| SHAP latency exceeds 2 s p95 per spec success criteria | Demo feels sluggish | Pre-build `TreeExplainer` at startup (Task 9 sub-step). If still slow, cap batch size at 20 instead of 100. |
| Pydantic validation errors don't produce the documented `ErrorResponse` shape | OpenAPI lies to consumers about error format | Task 11 wires a custom `RequestValidationError` handler that returns the standard `ErrorResponse` shape. Verified by `test_errors.py`. |
| Day-job interruption mid-slice leaves the codebase broken | Lost evening when picking back up | Every slice ends with a green `pytest` + a commit. No WIP commits, no half-implemented endpoints. |

---

## Task List

### Phase 1: Foundation (Tasks 1–5)

Goal: working FastAPI app with `/health`, predictor wired in, test infrastructure ready. No business endpoints yet.

#### Task 1: Add backend dependencies and install

**Description:** Add `fastapi`, `uvicorn[standard]`, `pytest`, `httpx` to `requirements.txt`. Install into `./venv`. Confirm `verify_model.py` still passes (no version conflicts).

**Acceptance criteria:**
- [ ] `requirements.txt` has the 4 new packages with version bands matching spec
- [ ] `pip install -r requirements.txt` exits 0
- [ ] `python verify_model.py` still passes all 6 steps

**Verification:** `pip install -r requirements.txt && python verify_model.py`

**Dependencies:** None
**Files:** `requirements.txt`
**Scope:** XS

---

#### Task 2: App skeleton + `/health` endpoint + CORS dev stub

**Description:** Create the `app/` package with `main.py` (FastAPI factory, CORS middleware, lifespan placeholder), `routers/health.py` (returns `{"status": "ok", "model": "xgboost-v7-seed0"}`), and the package `__init__.py` files. Configure uvicorn to log at `warning` level by default (no request bodies in logs).

**Acceptance criteria:**
- [ ] `uvicorn app.main:app` starts without error
- [ ] `GET /health` returns 200 with the documented payload
- [ ] `GET /docs` renders Swagger UI listing `/health`
- [ ] CORS preflight from `http://localhost:5173` succeeds (returns appropriate `Access-Control-Allow-Origin`)
- [ ] No request body content appears in stdout/stderr at default log level

**Verification:** `uvicorn app.main:app --port 8000 &` then `curl -i http://127.0.0.1:8000/health` and `curl -i -X OPTIONS http://127.0.0.1:8000/health -H "Origin: http://localhost:5173" -H "Access-Control-Request-Method: GET"`

**Dependencies:** Task 1
**Files:** `app/__init__.py`, `app/main.py`, `app/routers/__init__.py`, `app/routers/health.py`
**Scope:** S

---

#### Task 3: Predictor singleton via dependency injection

**Description:** Create `app/dependencies.py` exposing `get_predictor()` — an `@lru_cache(maxsize=1)`-wrapped factory that builds and returns one `ReadmissionPredictor` per process. Wire `lifespan` in `main.py` to pre-warm it at startup (so the first real request doesn't pay the 3 s parquet load).

**Acceptance criteria:**
- [ ] `get_predictor()` returns the same instance across N calls
- [ ] Startup log shows predictor pre-warmed (single "Predictor ready" line)
- [ ] App startup time < 10 s on the dev machine

**Verification:** `python -c "from app.dependencies import get_predictor; a, b = get_predictor(), get_predictor(); assert a is b"` plus visual confirmation of startup time

**Dependencies:** Task 2
**Files:** `app/dependencies.py`, `app/main.py` (edit `lifespan`)
**Scope:** XS

---

#### Task 4: Pydantic schemas for input + responses + errors

**Description:** Create `app/schemas/{features,responses,errors}.py`. `features.py` defines `PatientFeatures` with all 50 fields generated from `src/schema.py:FEATURE_COLS` — numeric fields typed `float | None`, categorical fields typed `str`, with `ConfigDict(extra="forbid")`. `responses.py` defines `PredictionResponse`, `BatchPredictionResponse`, `ExplanationResponse`, `MetadataResponse`. `errors.py` defines `ErrorResponse` and the `APIError` enum (`VALIDATION_ERROR`, `NOT_FOUND`, `INTERNAL_ERROR`).

**Acceptance criteria:**
- [ ] `PatientFeatures` has exactly 50 fields, named identically to `FEATURE_COLS`
- [ ] `PatientFeatures(**valid_row_dict)` succeeds
- [ ] `PatientFeatures(**{**valid_row_dict, "surgeon_name": "X"})` raises a Pydantic `ValidationError`
- [ ] All response models have `Field(..., description=...)` on every field (OpenAPI docs are usable)

**Verification:** `python -c "from app.schemas.features import PatientFeatures; assert len(PatientFeatures.model_fields) == 50"`

**Dependencies:** Task 1
**Files:** `app/schemas/__init__.py`, `app/schemas/features.py`, `app/schemas/responses.py`, `app/schemas/errors.py`
**Scope:** S

---

#### Task 5: Test infrastructure (conftest + fixtures)

**Description:** Create `tests/__init__.py` and `tests/conftest.py`. The conftest exposes: (a) `client` — a session-scoped `TestClient(app)` with the predictor pre-warmed; (b) `sample_patients` — a session-scoped fixture that loads 20 rows from the V7 parquet via `df.sample(n=20, random_state=42)`; (c) `valid_patient_dict` — a function-scoped fixture returning one row from `sample_patients` as a dict with only the 50 feature columns.

**Acceptance criteria:**
- [ ] `pytest tests/` (with no test files yet) exits 0 (collects 0, no errors)
- [ ] `pytest --collect-only` shows the fixtures are discovered
- [ ] Adding a smoke test (`def test_smoke(client): assert client.get("/health").status_code == 200`) passes

**Verification:** Run the smoke test, then delete it: `pytest tests/test_smoke.py -v` then `rm tests/test_smoke.py`

**Dependencies:** Tasks 2, 3, 4
**Files:** `tests/__init__.py`, `tests/conftest.py`
**Scope:** S

---

### ✅ Checkpoint 1 — Foundation complete

Before proceeding to Phase 2:
- [ ] `uvicorn app.main:app` starts in < 10 s and serves `/docs`
- [ ] `pytest` runs to completion with 0 failures (0 tests is OK at this stage)
- [ ] `python verify_model.py` still passes
- [ ] Git log shows 5 atomic commits (one per task)

---

### Phase 2: Endpoints (Tasks 6–10, one vertical slice per task)

Goal: all 5 business endpoints implemented and tested, including the 6 non-negotiable feature-alignment tests.

#### Task 6: `GET /metadata` + `test_metadata.py`

**Description:** Implement `app/routers/metadata.py` returning a `MetadataResponse` with: feature names + per-feature type (numeric/categorical), per-categorical the list of known levels (from the fitted `LabelEncoder.classes_`), per-numeric the min/median/max/pct_nan from `missing_data_profile_v7.csv`, model info (name, seed, published AUROC), and the default threshold (0.5). Wire into `main.py`. Write `test_metadata.py` with happy-path test + assertion that all 50 feature names appear.

**Acceptance criteria:**
- [ ] `GET /metadata` returns 200 with all 50 features described
- [ ] Each categorical feature has its `levels` field populated from `LabelEncoder.classes_`
- [ ] Each numeric feature has min/median/max/pct_nan from the missing-data profile
- [ ] `test_metadata.py` passes

**Verification:** `pytest tests/test_metadata.py -v`

**Dependencies:** Tasks 3, 4, 5
**Files:** `app/routers/metadata.py`, `app/main.py` (include router), `tests/test_metadata.py`
**Scope:** S

---

#### Task 7: `POST /predictions` (single) + the 6 non-negotiable alignment tests + `test_predictions.py`

**Description:** Implement `app/routers/predictions.py` with `POST /predictions` accepting `PatientFeatures`, query param `threshold: float = 0.5`, returning `PredictionResponse`. Inside the handler: convert `PatientFeatures` → 1-row DataFrame, call `predictor.predict_proba(df)[0]`, build response. Write `test_predictions.py` (happy path, threshold variation, bad input) AND `test_feature_alignment.py` with all 6 non-negotiable tests from `spec.md` §Testing Strategy.

**Acceptance criteria:**
- [ ] `POST /predictions` returns 200 with valid probability in [0, 1]
- [ ] `test_predictions.py` passes (happy path + threshold variation)
- [ ] **All 6 tests in `test_feature_alignment.py` pass** (column shuffle, categorical encoding, missing field, extra field, unseen categorical, end-to-end AUROC equivalence)
- [ ] p95 latency for single prediction < 200 ms (verified with simple loop)

**Verification:** `pytest tests/test_predictions.py tests/test_feature_alignment.py -v && python -c "from fastapi.testclient import TestClient; from app.main import app; import time; c = TestClient(app); from tests.conftest import _load_sample; row = _load_sample(); c.post('/predictions', json=row); t=time.perf_counter(); [c.post('/predictions', json=row) for _ in range(50)]; print(f'avg: {(time.perf_counter()-t)/50*1000:.1f}ms')"`

**Dependencies:** Tasks 3, 4, 5
**Files:** `app/routers/predictions.py`, `app/main.py` (include router), `tests/test_predictions.py`, `tests/test_feature_alignment.py`
**Scope:** M

---

#### Task 8: `POST /predictions/batch` (multi-patient) + tests

**Description:** Add `POST /predictions/batch` to `app/routers/predictions.py`. Accepts `{"patients": [PatientFeatures, ...]}` with max 100 patients, returns `BatchPredictionResponse` with per-patient probability + prediction. Enforce the 100-row cap with HTTP 422 if exceeded. Extend `test_predictions.py`.

**Acceptance criteria:**
- [ ] Batch of 5 patients returns 5 probabilities matching individual calls (to 1e-9)
- [ ] Batch of 101 patients returns HTTP 422 with `code: "VALIDATION_ERROR"`
- [ ] Empty batch (`[]`) returns HTTP 422

**Verification:** `pytest tests/test_predictions.py -v -k batch`

**Dependencies:** Task 7
**Files:** `app/routers/predictions.py`, `tests/test_predictions.py`
**Scope:** S

---

#### Task 9: `POST /explanations` (SHAP) + `app/serialization.py` + `test_explanations.py`

**Description:** Implement `app/serialization.py` with `numpy_to_json_safe(arr)` helper that converts `np.ndarray` → nested list of Python floats (handles NaN → `None`). Implement `POST /explanations` accepting a single `PatientFeatures`, returning `ExplanationResponse` with: `shap_values` (length 50 list of floats), `base_value` (float), `feature_names` (length 50 list), `feature_values_transformed` (length 50 list — the post-encoding values that went into the model, for waterfall chart context). Pre-build the TreeExplainer at startup (extend `lifespan` in `main.py`).

**Acceptance criteria:**
- [ ] `POST /explanations` returns 200 with `shap_values` length 50, `feature_names` length 50, `base_value` is a float
- [ ] `sum(shap_values) + base_value` is finite and close to the model's logit output for the row
- [ ] p95 latency < 2 s for single explanation
- [ ] `test_explanations.py` passes

**Verification:** `pytest tests/test_explanations.py -v`

**Dependencies:** Tasks 3, 4, 5
**Files:** `app/routers/predictions.py` (add /explanations) OR new `app/routers/explanations.py`, `app/serialization.py`, `app/main.py` (extend lifespan), `tests/test_explanations.py`
**Scope:** M

---

#### Task 10: `GET /examples?n=N` (ID-stripped rows) + PHI-guard test

**Description:** Implement `app/routers/examples.py`. Endpoint accepts `n: int = Query(default=5, ge=1, le=100)`. Reads from V7 parquet (via a session-cached helper), takes `df.sample(n=n, random_state=...)`, **drops all 5 columns in `src.schema.ID_COLS`** plus the `TARGET_COL`, returns the remaining 50 feature columns as a list of dicts. Write `test_examples.py` asserting (a) response length matches `n`, (b) every returned object has exactly the 50 feature column keys, (c) **none** of the 5 `ID_COLS` keys appear anywhere in the response.

**Acceptance criteria:**
- [ ] `GET /examples?n=5` returns 5 objects
- [ ] Every object has all 50 feature keys
- [ ] **No object contains `subject_id`, `hadm_id`, `admittime_dt`, `dischtime_dt`, or `insurance` as a key** (PHI guard — test failure stops the build)
- [ ] `n=101` returns HTTP 422
- [ ] `n=0` returns HTTP 422

**Verification:** `pytest tests/test_examples.py -v`

**Dependencies:** Tasks 3, 4, 5
**Files:** `app/routers/examples.py`, `app/main.py` (include router), `tests/test_examples.py`
**Scope:** S

---

### ✅ Checkpoint 2 — All endpoints functional

Before proceeding to Phase 3:
- [ ] All 5 endpoints return valid responses for happy-path requests
- [ ] All 6 feature-alignment tests pass
- [ ] `test_examples.py` PHI guard passes
- [ ] Full `pytest` suite < 60 s
- [ ] Manual smoke: `curl` each endpoint and visually verify the response

---

### Phase 3: Polish (Tasks 11–14)

Goal: unified error model, polished OpenAPI, README updated, final verification.

#### Task 11: Unified error model + `test_errors.py`

**Description:** Add a `RequestValidationError` exception handler to `app/main.py` that returns the `ErrorResponse` shape from `app/schemas/errors.py` instead of FastAPI's default. Add a generic `Exception` handler returning `{code: "INTERNAL_ERROR"}` (never leaking the actual exception text). Write `test_errors.py` verifying every documented 4xx/5xx code is reachable and returns the documented shape.

**Acceptance criteria:**
- [ ] All 422 responses match the `ErrorResponse` schema exactly
- [ ] 500 responses return `{"error": {"code": "INTERNAL_ERROR", "message": "Internal server error"}}` with no traceback leaked
- [ ] `test_errors.py` covers each documented code (400-class minimum: 422; 500-class: 500)

**Verification:** `pytest tests/test_errors.py -v`

**Dependencies:** Tasks 4, 7 (need a working endpoint to trigger errors against)
**Files:** `app/main.py` (handlers), `tests/test_errors.py`
**Scope:** S

---

#### Task 12: OpenAPI polish

**Description:** Walk every router and add: `tags=["..."]` on `APIRouter`, `summary=` and `description=` on every endpoint, `response_model=` on every endpoint, `responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}` so Swagger documents error shapes. Add one realistic `examples=` to `PatientFeatures` (one sanitized row from the test fixture).

**Acceptance criteria:**
- [ ] `/docs` shows every endpoint grouped by tag with descriptions
- [ ] Every endpoint documents its 422 + 500 response shape
- [ ] `PatientFeatures` schema shows a complete example payload in `/docs`
- [ ] Manual: open `/docs`, click "Try it out" on `/predictions`, verify the example pre-fills

**Verification:** `python -c "from app.main import app; import json; s = app.openapi(); assert all('description' in op for path in s['paths'].values() for op in path.values() if isinstance(op, dict) and 'description' in op or 'summary' in op)"` + manual /docs check

**Dependencies:** Tasks 6, 7, 8, 9, 10
**Files:** all `app/routers/*.py`, `app/schemas/features.py` (add example)
**Scope:** S

---

#### Task 13: README "Running the backend" section

**Description:** Append a "Running the backend" section to `README.md` documenting: how to activate venv, how to start uvicorn, where `/docs` lives, how to run the test suite, the 5 endpoints with one-line descriptions.

**Acceptance criteria:**
- [ ] Section reads top-to-bottom as a runnable demo guide
- [ ] All commands listed are copy-pasteable PowerShell
- [ ] Links to `spec.md` and `plan.md` for context

**Verification:** Visual review

**Dependencies:** Task 12 (so endpoint list is final)
**Files:** `README.md`
**Scope:** XS

---

#### Task 14: Final verification + demo dry-run

**Description:** Run the full success-criteria checklist from `spec.md`. Run `verify_model.py` one more time. Run full `pytest` suite with timing. Manual curl smoke for all 5 endpoints. Measure single-prediction p95 latency over 100 calls and single-explanation p95 over 20 calls. Generate and inspect `openapi.json`. Fix any issues surfaced.

**Acceptance criteria:**
- [ ] All 10 spec.md Success Criteria checkboxes green
- [ ] No request body or feature value in any log line at any level
- [ ] `verify_model.py` still passes
- [ ] Git log: 14 atomic commits (one per task) + spec + plan + initial = 17 total

**Verification:** Manual checklist + `pytest -v` + `python verify_model.py` + measured latencies

**Dependencies:** All previous
**Files:** Possibly small fixes anywhere
**Scope:** S

---

### ✅ Checkpoint 3 — Ready for Saturday demo

- [ ] Every Success Criteria from `spec.md` green
- [ ] Backend starts in one command, demos in one browser tab
- [ ] No PHI leak path remains (audited via `test_examples.py` + log review + `git status`)
- [ ] Frontend stack pick + wire-up can begin Saturday morning

---

## Sizing summary

| Phase | Tasks | Total scope |
|---|---|---|
| 1 Foundation | 5 | 4×S + 1×XS = ~4 hrs |
| 2 Endpoints | 5 | 2×M + 3×S = ~8 hrs |
| 3 Polish | 4 | 3×S + 1×XS = ~3 hrs |
| **Total** | **14** | **~15 hrs** — fits the 5-day weeknight + Saturday-morning window |

## What's NOT in this plan

- Frontend stack decision and build (Saturday morning, separate plan)
- Containerization / Docker (out of scope per spec non-goals)
- GitHub remote setup (deferred until build is stable per spec Boundaries)
- Threshold-calibration analysis (using spec-decided default of 0.5)
- Model retraining or alternative model exploration (frozen per spec)
- Performance optimization beyond the spec latency targets

---

*End of plan. Awaiting approval before starting Task 1.*
