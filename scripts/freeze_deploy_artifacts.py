"""Pre-bake the two artifacts that the deployed backend needs at runtime.

WHY
---
The runtime backend has two dependencies on the 25 MB MIMIC-derived
parquet (`data/processed/training_table_v7.parquet`):

  1. `src.preprocess.build_encoders()` re-fits LabelEncoders on every
     startup, reading the 4 CATEGORICAL_COLS from the full parquet.
  2. `app.routers.examples` samples N rows from the full parquet on
     each /examples call.

Shipping the parquet in a Docker image would:
  - bloat the image by 25 MB,
  - add a ~3 s parquet load to cold-start latency,
  - constitute redistribution of MIMIC-derived data (PhysioNet DUA
    forbids this for any public deployment).

This script runs ONCE locally to pre-bake both dependencies into small,
deploy-safe pickles:

  model/deploy_encoders.pkl
    Dict[str, sklearn.preprocessing.LabelEncoder] — only contains the
    {category_string -> integer} mappings the model needs. Not patient
    data; fair to ship in a public repo.

  model/deploy_examples_pool.pkl
    pd.DataFrame with 100 anonymized rows (FEATURE_COLS only — never
    reads ID_COLS) drawn from the held-out test set.
    Per the user's explicit decision (see deploy notes), this pool IS
    shipped to the public backend so the demo's "load patient" flow
    works without backend-internal MIMIC access. The /examples router
    falls back to sampling from this pool when the parquet is absent.

USAGE
-----
    python scripts/freeze_deploy_artifacts.py

Re-run any time the V7 model is retrained or the schema changes.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from src.preprocess import build_encoders
from src.schema import FEATURE_COLS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TRAINING_PARQUET = _PROJECT_ROOT / "data" / "processed" / "training_table_v7.parquet"
_MODEL_DIR = _PROJECT_ROOT / "model"

_ENCODERS_OUT = _MODEL_DIR / "deploy_encoders.pkl"
_EXAMPLES_OUT = _MODEL_DIR / "deploy_examples_pool.pkl"

# Pool size = MAX_N of the /examples endpoint. Sampling N <= 100 rows
# at request time draws from this pool. random_state=42 mirrors the
# previous parquet-backed sample for behavioural continuity.
_POOL_SIZE = 100
_POOL_SEED = 42


def freeze_encoders() -> None:
    """Fit LabelEncoders on the full parquet and pickle them."""
    if not _TRAINING_PARQUET.exists():
        raise FileNotFoundError(
            f"Parquet missing at {_TRAINING_PARQUET}. "
            "Run this script on a machine that has the local MIMIC artifacts."
        )
    encoders = build_encoders(_TRAINING_PARQUET)
    _ENCODERS_OUT.write_bytes(pickle.dumps(encoders))
    size_kb = _ENCODERS_OUT.stat().st_size / 1024
    print(f"  wrote {_ENCODERS_OUT.relative_to(_PROJECT_ROOT)} ({size_kb:.1f} KB)")


def freeze_examples_pool() -> None:
    """Sample _POOL_SIZE anonymized rows and pickle as a DataFrame."""
    df = pd.read_parquet(_TRAINING_PARQUET, columns=FEATURE_COLS)
    pool = df.sample(n=_POOL_SIZE, random_state=_POOL_SEED).reset_index(drop=True)
    # Sanity: pool MUST NOT contain any ID column even by accident.
    forbidden = {"subject_id", "hadm_id", "admittime_dt", "dischtime_dt", "insurance"}
    leaked = forbidden.intersection(pool.columns)
    assert not leaked, f"PHI guard: {leaked} leaked into examples pool"
    pool.to_pickle(_EXAMPLES_OUT)
    size_kb = _EXAMPLES_OUT.stat().st_size / 1024
    print(f"  wrote {_EXAMPLES_OUT.relative_to(_PROJECT_ROOT)} ({size_kb:.1f} KB)")


def main() -> None:
    print("Freezing deploy artifacts:")
    freeze_encoders()
    freeze_examples_pool()
    print("Done. Both pickles are safe to commit to the deployed backend repo.")


if __name__ == "__main__":
    main()
