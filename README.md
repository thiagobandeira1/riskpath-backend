# Hospital Predictive Model App

Local-only working folder for building a web application on top of the
**V17-validated deployed XGBoost** 30-day Medicare hospital readmission
prediction model.

This is a self-contained copy of the trained XGBoost model, the V7 50-feature
training table, the raw MIMIC-IV v3.1 parquet snapshot (for any future
feature re-engineering), a small inference + SHAP module, and a verification
script. The source code that produced the model lives in the publication repo
at `medicare-30day-readmission-mimic-iv` and is **not** included here.

## What lives where

```
Hospital Predictive Model App/
├── model/                                # Trained model + reference artefacts
│   ├── v7_seed0_xgboost.pkl              #   Deployed XGBoost (seed 0, 0.7929 test AUROC)
│   ├── v7_feature_cols.json              #   50-feature column order for input alignment
│   ├── v7_split_indices.npz              #   Patient-grouped train/val/test indices
│   ├── v7_summary.json                   #   Published 10-seed test AUROC reference
│   ├── v7_feature_importance_current_run.csv
│   └── missing_data_profile_v7.csv       #   Per-column NaN + zero-share profile
│
├── src/                                  # Inference package
│   ├── schema.py                         #   FEATURE_COLS / CATEGORICAL_COLS / TARGET_COL
│   ├── preprocess.py                     #   LabelEncoder fitting + transform
│   └── inference.py                      #   ReadmissionPredictor class (predict + SHAP)
│
├── data/
│   ├── raw/                              # Raw MIMIC-IV v3.1 parquets (~5.1 GB, 66 files)
│   └── processed/
│       └── training_table_v7.parquet     # 244,576 admissions x 56 columns (V7 features)
│
├── notebooks/
│   └── Capstone_Final_Notebook.ipynb     # Reference notebook (for design context)
│
├── app/                                  # FastAPI backend (see "Running the backend" below)
│   ├── main.py                           #   App factory + CORS + lifespan
│   ├── dependencies.py                   #   Predictor singleton
│   ├── error_handlers.py                 #   Unified 422 / 500 → ErrorResponse
│   ├── serialization.py                  #   numpy → JSON helpers
│   ├── routers/                          #   /health, /metadata, /predictions[/batch],
│   │                                     #     /explanations, /examples
│   └── schemas/                          #   Pydantic features / responses / errors
│
├── tests/                                # pytest suite (61 tests, ~4 s)
│   ├── conftest.py                       #   Fixtures: client, sample_patients
│   ├── test_feature_alignment.py         #   6 NON-NEGOTIABLE alignment gates
│   ├── test_examples.py                  #   PHI guard + endpoint contract
│   ├── test_explanations.py              #   SHAP shape + additivity
│   ├── test_metadata.py                  #   Feature schema + model info
│   ├── test_predictions.py               #   Single + batch
│   └── test_errors.py                    #   422 / 500 envelope shape + traceback no-leak
│
├── verify_model.py                       # End-to-end inference + SHAP sanity check
├── spec.md                               # Backend spec (the gated source of truth)
├── plan.md                               # 14-task implementation plan + checkpoints
├── requirements.txt
├── .gitignore                            # Excludes data/ so MIMIC never leaks if you init git
└── README.md                             # (this file)
```

## Why XGBoost only

The deployed model selected for the publication is XGBoost (continuity with
the defended original capstone protocol). LightGBM scores marginally higher
on the held-out test set under the V17 no-leakage protocol (LightGBM 0.7979
vs XGBoost 0.7935, 10-seed averages), but the gap is well inside the 5-fold
CV standard deviation (~0.003), so the two models are statistically
indistinguishable. To keep this app's dependency tree slim, only the deployed
model and the libraries it needs are pinned in `requirements.txt`.

## Seed strategy — single seed-0 XGBoost

The published headline AUROC of **0.7935** is a **10-seed average**. Only the
seed-0 pickle was serialised in the source repo; reproducing the 10-seed
average would require retraining and shipping the other 9 seeds. This app
**ships seed 0 only**, which scores **0.7929** on the same full test set — a
gap of −0.0006 AUROC, well within validation noise.

The reasoning for staying on a single seed:

- The 0.0006 AUROC gap is an order of magnitude smaller than the 5-fold CV
  standard deviation (≈0.003). For all clinical-deployment purposes the two
  numbers carry the same information.
- A single-model pipeline is faster (one forward pass per patient), simpler
  to debug, easier to serve, and produces a single coherent SHAP attribution
  per prediction. 10-seed averaging would require averaging SHAP across the
  ensemble — defensible but non-standard.
- The web app is a publication artefact + immigration evidence artefact, not
  a benchmark-leaderboard submission. Stability across seeds is already
  documented in `model/v7_summary.json` and in §10.3c of the notebook.
- If the headline 0.7935 is needed for any later regulatory filing, the
  10-seed average can be reconstructed offline by replaying the training
  script in the publication repo (see `src/run_v7_ensemble.py` there) and
  serialising every seed. The pipeline is fully reproducible.

The published `v7_summary.json` is shipped alongside the model so the
deployed-seed number (0.7929) and the academic headline number (0.7935) are
both visible and can be cited honestly.

## Data provenance and licensing

The parquet files under `data/` are derived from **MIMIC-IV v3.1** (PhysioNet).
MIMIC-IV is distributed under the [PhysioNet Credentialed Health Data Use
Agreement](https://physionet.org/news/post/415). Key constraints:

- The data is for **local development only**.
- The data must **not** be redistributed (no upload to GitHub, no email, no
  cloud bucket without the appropriate DUA in place).
- `.gitignore` excludes `data/` to prevent accidental commits. If you ever
  `git init` this folder, verify the exclusion is still in effect with
  `git status` before your first add.

`data/raw/` contains the full 66-file MIMIC-IV parquet snapshot (~5.1 GB) so
any future feature re-engineering can be done entirely inside this folder
without reaching back to the original capstone workspace.

## Quick start

```powershell
# 1. Create + activate a virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Verify the model loads, predicts, and produces SHAP values correctly
python verify_model.py
```

`verify_model.py` will:

1. Load the V7 training parquet and the saved patient-grouped split indices.
2. Load the deployed seed-0 XGBoost pickle.
3. Run inference on the full held-out test set (49,191 admissions).
4. Confirm test AUROC matches the published 10-seed value in
   `model/v7_summary.json` to within ±0.01 (typical gap is −0.0006).
5. Compute SHAP values on a 50-row sample to confirm interpretability works.

If any step fails the script exits non-zero and prints the cause.

## Programmatic use

```python
from src.inference import ReadmissionPredictor
import pandas as pd

# Load patients (any DataFrame with the 50 V7 feature columns will work)
df = pd.read_parquet("data/processed/training_table_v7.parquet").head(10)

predictor = ReadmissionPredictor()             # XGBoost, seed 0

probs = predictor.predict_proba(df)            # shape: (10,)
labels = predictor.predict(df, threshold=0.5)  # shape: (10,)

# SHAP explanations for the same rows
shap_out = predictor.explain(df)
# shap_out["shap_values"]   -> (10, 50) per-feature contributions to class 1
# shap_out["base_value"]    -> model expected value (log-odds)
# shap_out["feature_names"] -> list of 50 feature names in column order
```

## Reproducibility notes

- All model artefacts and the V7 parquet were copied directly from commit
  `cef0d9c` of `thiagobandeira1/Medicare-30day-Readmission-MIMIC-IV` on
  2026-05-17.
- The LabelEncoders used at training time are not serialised in the original
  repo. `src/preprocess.build_encoders()` re-fits them from the full V7 parquet
  (244,576 rows), which guarantees every observed category is covered and the
  encoding is identical to the one used during training.
- Pinned package versions in `requirements.txt` match the source repo's
  Conda environment for the XGBoost path only (Python 3.12.10, scikit-learn
  1.8.0, XGBoost 3.2.0, SHAP 0.51.0).

## Running the backend

The FastAPI backend wraps `src.inference.ReadmissionPredictor` behind 5 HTTP
endpoints. Design + scope live in [`spec.md`](spec.md); the implementation
plan lives in [`plan.md`](plan.md).

```powershell
# 1. Activate the venv (every new shell)
.\venv\Scripts\Activate.ps1

# 2. Start the dev server with hot-reload on file change
uvicorn app.main:app --reload --port 8000

# Browse:
#   http://127.0.0.1:8000/docs            # Swagger UI — try-it-out enabled
#   http://127.0.0.1:8000/openapi.json    # Machine-readable schema
#   http://127.0.0.1:8000/health          # Liveness probe
```

Startup is ~2 s (model load + LabelEncoder fit + SHAP TreeExplainer build,
all pre-warmed via `lifespan`).

### Endpoints

| Method | Path                  | Purpose |
|--------|-----------------------|---------|
| GET    | `/health`             | Liveness — returns model identifier, no model load |
| GET    | `/metadata`           | Feature schema + categorical levels + numeric stats + model info |
| GET    | `/examples?n=N`       | N anonymized rows (1-100) ready to POST. ID columns never read into memory. |
| POST   | `/predictions`        | Single patient → probability + binary prediction (`?threshold=0.5` query param) |
| POST   | `/predictions/batch`  | Up to 100 patients per request, single XGBoost forward pass |
| POST   | `/explanations`       | SHAP contributions per feature (log-odds) + base value + probability |

Every 4xx/5xx response uses the same `ErrorResponse` envelope:
`{"error": {"code": "VALIDATION_ERROR" | "INTERNAL_ERROR", "message": "...", "details": {...}}}`.

### Running the test suite

```powershell
# Full suite (~4 s, 58 tests + 3 intentional skips)
pytest tests/ -v

# Just the 6 non-negotiable feature-alignment gates
pytest tests/test_feature_alignment.py -v

# Re-verify model layer hasn't regressed (run before each demo)
python verify_model.py
```

### Pre-demo checklist

Before showing the backend to anyone:

```powershell
python verify_model.py             # 1. Model still scores at AUROC 0.7929
pytest tests/                       # 2. All 58 tests green, including the 6 alignment gates
uvicorn app.main:app --port 8000    # 3. Dev server up, /docs reachable
```

If any of those three fail, **do not demo**.

## Status

**This folder is git-initialized locally (no remote).** All MIMIC-derived
data in `data/` is gitignored. Before adding a GitHub remote, re-audit
`.gitignore` and run `git status` to confirm no patient data has been
accidentally staged.

Built 2026-05-17 from the publication repo's V17-validated artefacts.
Backend built 2026-05-17 against `spec.md` / `plan.md`.
