"""Curate a hand-picked, ordered demo patient pool for the /examples endpoint.

WHY
---
The original deploy_examples_pool.pkl was a *random* 100-row sample of the
training data. For a live demo that read poorly:
  - Most randomly sampled patients are low risk (the outcome is rare), so the
    cards didn't show a meaningful risk spread.
  - Some sampled patients had a "DIED" discharge disposition — a confusing
    demo case, because a deceased patient cannot be readmitted.

This script replaces that pool with a curated, ORDERED set of five patients
spanning the full risk spectrum, all with clean, intuitive discharge
dispositions. The /examples endpoint returns them in order (pool.head(n)),
so the demo cards read left-to-right from "not that bad" to "really bad."

Target spectrum (probability midpoints):
    #1  ~0.02  VERY LOW   — the reassuring baseline
    #2  ~0.10  LOW
    #3  ~0.38  MODERATE   — crosses into intervention territory
    #4  ~0.68  HIGH
    #5  ~0.90  VERY HIGH  — most likely to return; the dramatic case

Clean dispositions only (no DIED, HOSPICE, Unknown, AGAINST ADVICE, PSYCH).

DUA NOTE
--------
These are still MIMIC-derived rows (FEATURE_COLS only, no identifiers), the
same posture the user already approved for public deployment. We are merely
choosing *which* anonymized rows to surface, not exposing anything new.

USAGE
    python scripts/curate_demo_patients.py
    # overwrites model/deploy_examples_pool.pkl with the curated 5-row pool
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.inference import ReadmissionPredictor
from src.schema import FEATURE_COLS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TRAINING_PARQUET = _PROJECT_ROOT / "data" / "processed" / "training_table_v7.parquet"
_POOL_OUT = _PROJECT_ROOT / "model" / "deploy_examples_pool.pkl"

# Total pool size. The /examples endpoint allows n up to 100; the pool must be
# at least that large so head(n) can satisfy any request. Only the first 5 are
# the curated demo spectrum — the rest are clean-disposition backfill.
_POOL_SIZE = 100
_FILLER_SEED = 42

# Discharge dispositions that read well in a demo. Excludes DIED / HOSPICE
# (terminal — can't be readmitted / awkward), Unknown (looks like missing
# data), and AGAINST ADVICE / PSYCH FACILITY (potentially distracting).
_CLEAN_DISPOSITIONS = {
    "HOME",
    "HOME HEALTH CARE",
    "SKILLED NURSING FACILITY",
    "REHAB",
}

# (target_probability, band_label, preferred_disposition_or_None).
# Preferred disposition steers the narrative (low-risk patients tend to go
# home; high-risk patients tend to need a higher level of care) but is a
# soft preference — we fall back to any clean disposition if needed.
_TARGETS = [
    (0.02, "VERY LOW", "HOME"),
    (0.10, "LOW", "HOME HEALTH CARE"),
    (0.38, "MODERATE", "REHAB"),
    (0.68, "HIGH", "SKILLED NURSING FACILITY"),
    (0.90, "VERY HIGH", "SKILLED NURSING FACILITY"),
]


def main() -> None:
    if not _TRAINING_PARQUET.exists():
        raise FileNotFoundError(
            f"Parquet missing at {_TRAINING_PARQUET}. Run on a machine with the "
            "local MIMIC artifacts."
        )

    df = pd.read_parquet(_TRAINING_PARQUET, columns=FEATURE_COLS)
    predictor = ReadmissionPredictor()
    df = df.assign(_proba=predictor.predict_proba(df))

    clean = df[df["discharge_location"].isin(_CLEAN_DISPOSITIONS)].copy()

    chosen_idx: list[int] = []
    chosen_rows: list[pd.Series] = []
    for target, label, pref_dispo in _TARGETS:
        # Prefer the preferred disposition; fall back to all clean rows.
        candidates = clean[clean["discharge_location"] == pref_dispo]
        if candidates.empty:
            candidates = clean
        # Exclude already-chosen rows so we never pick the same patient twice.
        candidates = candidates[~candidates.index.isin(chosen_idx)]
        # Pick the row whose probability is closest to the target.
        pick_idx = (candidates["_proba"] - target).abs().idxmin()
        chosen_idx.append(pick_idx)
        chosen_rows.append(df.loc[pick_idx])

    stars = pd.DataFrame(chosen_rows).reset_index(drop=True)

    # Report the curated demo stars (cards #1-#5) for the demo script.
    print("Curated demo patients (cards #1-#5, low -> high risk):")
    for i, (target, label, _pref) in enumerate(_TARGETS):
        row = stars.iloc[i]
        print(
            f"  #{i + 1}  proba={row['_proba']:.3f}  band={label:9s}  "
            f"DRG={row['drg_code']!s:>5}  dx_chapter={row['primary_dx_chapter']!s:>3}  "
            f"dispo={row['discharge_location']!s:<24}  "
            f"prior_admits_6m={row['prior_admissions_6m']!s}"
        )

    # Backfill the pool to _POOL_SIZE so the /examples endpoint still satisfies
    # requests up to n=100 (and the existing API contract / tests). The demo
    # stars stay FIRST so head(5) == the curated spectrum; the remainder are
    # additional CLEAN-disposition rows (still no DIED/HOSPICE) for larger
    # requests, which the 5-card demo never surfaces anyway.
    filler = (
        clean[~clean.index.isin(chosen_idx)]
        .sample(n=_POOL_SIZE - len(stars), random_state=_FILLER_SEED)
    )
    pool = pd.concat([stars, filler], ignore_index=True)

    # PHI guard — belt and braces: no identifier column may sneak in.
    forbidden = {"subject_id", "hadm_id", "admittime_dt", "dischtime_dt", "insurance"}
    leaked = forbidden.intersection(pool.columns)
    assert not leaked, f"PHI guard: {leaked} leaked into curated pool"

    pool = pool.drop(columns=["_proba"])
    assert list(pool.columns) == list(FEATURE_COLS), "column drift vs FEATURE_COLS"

    pool.to_pickle(_POOL_OUT)
    size_kb = _POOL_OUT.stat().st_size / 1024
    print(
        f"\nWrote {_POOL_OUT.relative_to(_PROJECT_ROOT)} "
        f"({len(pool)} patients total; first 5 are the curated demo spectrum, "
        f"{size_kb:.1f} KB)"
    )


if __name__ == "__main__":
    main()
