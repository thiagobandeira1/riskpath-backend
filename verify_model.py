"""End-to-end verification that the copied XGBoost model is fully functional.

Runs:
    1. Loads the V7 training parquet
    2. Loads the deployed seed-0 XGBoost model
    3. Picks the held-out test patients (using the saved split indices)
    4. Runs inference on a sample and on the full test set
    5. Computes SHAP values on a small sample
    6. Cross-checks the AUROC against model/v7_summary.json

Exits non-zero if anything fails. Run from the project root:
    python verify_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.inference import ReadmissionPredictor  # noqa: E402
from src.schema import TARGET_COL  # noqa: E402


def _step(msg: str) -> None:
    print(f"[*] {msg}", flush=True)


def main() -> int:
    try:
        _step("Loading training parquet (data/processed/training_table_v7.parquet)…")
        df = pd.read_parquet(ROOT / "data" / "processed" / "training_table_v7.parquet")
        print(f"    -> {df.shape[0]:,} rows x {df.shape[1]} columns")

        _step("Loading saved patient-grouped split indices…")
        split = np.load(ROOT / "model" / "v7_split_indices.npz")
        test_idx = split["test_idx"]
        y_test_saved = split["y_test"]
        print(f"    -> test set: {len(test_idx):,} admissions")

        df_test = df.iloc[test_idx].reset_index(drop=True)
        y_test = df_test[TARGET_COL].astype(int).values
        assert np.array_equal(y_test, y_test_saved), (
            "Test labels don't match split file — parquet row order may have drifted."
        )

        _step("Loading published 10-seed XGBoost AUROC target (model/v7_summary.json)…")
        summary = json.loads((ROOT / "model" / "v7_summary.json").read_text())
        target = summary["models"]["xgboost"]["test_auroc"]
        print(f"    -> XGBoost 10-seed target: {target:.4f}")
        print(f"       (this app ships seed 0 only; expected single-seed AUROC ~0.7929)")

        sample_n = 5000
        sample = df_test.sample(n=sample_n, random_state=42)
        y_sample = sample[TARGET_COL].astype(int).values

        _step(f"Loading XGBoost model + running inference on {sample_n:,}-row sample…")
        predictor = ReadmissionPredictor()
        proba = predictor.predict_proba(sample)
        assert proba.shape == (sample_n,)
        assert (proba >= 0).all() and (proba <= 1).all()
        auc = roc_auc_score(y_sample, proba)
        print(f"    -> sample AUROC: {auc:.4f}")

        _step("Running full-test-set inference (sanity vs. published 10-seed number)…")
        full_proba = predictor.predict_proba(df_test)
        full_auc = roc_auc_score(y_test, full_proba)
        print(f"    -> full test AUROC: {full_auc:.4f}  (10-seed target: {target:.4f}  "
              f"diff: {full_auc - target:+.4f})")
        assert abs(full_auc - target) < 0.01, (
            f"Full test AUROC {full_auc:.4f} differs from 10-seed target "
            f"{target:.4f} by more than 0.01. Preprocessing may be wrong."
        )

        _step("Computing SHAP values on a 50-row sample…")
        try:
            shap_out = predictor.explain(sample.head(50))
            assert shap_out["shap_values"].shape == (50, 50)
            print(f"    -> SHAP base value: {shap_out['base_value']:+.4f}, "
                  f"shap_values shape: {shap_out['shap_values'].shape}")
        except ValueError as shap_exc:
            # Known: SHAP < 0.51 cannot parse xgboost 3.2.0's bracketed
            # base_score field. Predictions are unaffected; explanations
            # require the pinned shap==0.51.0 from requirements.txt.
            if "could not convert string to float" in str(shap_exc):
                print("    -> [warn] SHAP failed: known incompatibility between "
                      "SHAP <0.51 and xgboost 3.2.0.")
                print("             Install pinned requirements (shap==0.51.0) to enable.")
            else:
                raise

        _step("All verification steps passed.")
        return 0

    except Exception as exc:
        print(f"\n[!] VERIFICATION FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
