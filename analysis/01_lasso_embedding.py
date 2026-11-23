"""Platform-specific embedding-only Lasso probe.

The three canonical 384-dimensional embedding parquets use misleadingly raw-
sounding field names (``retweet_count``, ``useful_count``, and ``likes``), but
those fields already contain ``log1p(raw endorsement count)``.  This script
therefore uses each stored field directly.  Applying ``np.log1p`` again would
double-transform the outcome.

This is a complementary, leakage-resistant platform-specific analysis, not an
exact re-execution of the notebook's pooled headline.  For each platform it
uses one deterministic 80/20 split and selects the Lasso penalty by five-fold
cross-validation on the training portion only.

Final corrected results are intentionally checkpointed to a new file so that
the older double-transformed results and the preliminary fits whose selected
penalties hit the lower grid boundary cannot be mistaken for final output:

  results/lasso_results_stored_log1p_expanded_alpha.json

Run platforms sequentially on a resource-constrained laptop:

  python analysis/01_lasso_embedding.py --platform twitter
  python analysis/01_lasso_embedding.py --platform yelp
  python analysis/01_lasso_embedding.py --platform youtube

Completed platforms in the corrected checkpoint are skipped on rerun.
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

# Keep numerical backends single-threaded on the laptop.  These variables must
# be set before importing NumPy/scikit-learn.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_var] = "1"

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "embeddings"
OUTPUT_JSON = ROOT / "results" / "lasso_results_stored_log1p_expanded_alpha.json"
EMB = [str(i) for i in range(384)]

# These field names contain stored log1p counts, not raw counts.
FILES = {
    "twitter": ("embedding_endrosment_twitter.parquet", "retweet_count"),
    "yelp":    ("embedding_endrosment_yelp.parquet",     "useful_count"),
    "youtube": ("embedding_endrosment_youtube.parquet",  "likes"),
}

RANDOM_STATE = 0
ALPHAS = np.logspace(-6, 0, 50)


def load_platform(name):
    fn, count_col = FILES[name]
    path = DATA_DIR / fn
    if not path.exists():
        raise FileNotFoundError(path)
    cols = list(pq.ParquetFile(path).schema_arrow.names)
    missing = [column for column in EMB + [count_col] if column not in cols]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing[:5]}")
    df = pd.read_parquet(path, columns=EMB + [count_col])
    df = df.dropna(subset=EMB + [count_col])
    return df, count_col, path


def validate_stored_log1p(values, path, count_col):
    """Verify that expm1(stored value) recovers nonnegative integer counts."""
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(y) & (y >= 0)):
        raise ValueError(f"{path.name}:{count_col} has invalid stored outcomes")
    recovered = np.expm1(y)
    error = np.abs(recovered - np.rint(recovered))
    tolerance = 1e-7 * np.maximum(1.0, np.abs(recovered))
    if not np.all(np.isfinite(recovered) & (error <= tolerance)):
        raise ValueError(
            f"{path.name}:{count_col} does not look like log1p(integer count)"
        )
    return y, {
        "stored_min": float(np.min(y)),
        "stored_max": float(np.max(y)),
        "recovered_raw_count_max": float(np.max(recovered)),
        "max_integer_recovery_error": float(np.max(error)),
    }


def run_lasso(X, y, tag, count_col, validation):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    t0 = time.time()
    cv = LassoCV(cv=5, alphas=ALPHAS, random_state=RANDOM_STATE, n_jobs=1, precompute=False)
    cv.fit(Xtr, ytr)
    selected_at_boundary = bool(
        np.isclose(cv.alpha_, ALPHAS.min()) or np.isclose(cv.alpha_, ALPHAS.max())
    )
    if selected_at_boundary:
        raise RuntimeError(
            f"[{tag}] selected alpha={cv.alpha_:.9g} at the boundary of the "
            f"common grid [{ALPHAS.min():.1e}, {ALPHAS.max():.1e}]; extend the "
            "grid before reporting this platform."
        )
    r2 = r2_score(yte, cv.predict(Xte))
    nz = int(np.sum(cv.coef_ != 0))
    frac = nz / X.shape[1]
    print(f"[{tag}] n={X.shape[0]:>7d}  alpha={cv.alpha_:.8g}  "
          f"R2_test={r2:.4f}  nonzero={nz}/{X.shape[1]} ({frac:.2%})  "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return {
        "tag": tag,
        "n": int(X.shape[0]),
        "source_outcome_column": count_col,
        "target": "stored_log1p_endorsement_count",
        "outcome_transform_applied": "none",
        "split_seed": RANDOM_STATE,
        "test_size": 0.2,
        "alpha_grid_min": float(ALPHAS.min()),
        "alpha_grid_max": float(ALPHAS.max()),
        "alpha_grid_size": int(ALPHAS.size),
        "alpha_grid_label": "common_logspace_1e-6_to_1_50",
        "alpha": float(cv.alpha_),
        "alpha_selected_at_boundary": selected_at_boundary,
        "r2_test": float(r2),
        "nonzero": nz,
        "nonzero_frac": float(frac),
        "outcome_validation": validation,
    }


def load_existing_results():
    if OUTPUT_JSON.exists():
        with OUTPUT_JSON.open(encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(results):
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_JSON.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    temporary.replace(OUTPUT_JSON)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=["all", *FILES.keys()],
        default="all",
        help="Run one platform at a time (recommended) or all sequentially.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = load_existing_results()
    done_tags = {
        r["tag"]
        for r in results
        if r.get("target") == "stored_log1p_endorsement_count"
        and r.get("outcome_transform_applied") == "none"
    }
    platforms = list(FILES) if args.platform == "all" else [args.platform]
    for name in platforms:
        if name in done_tags:
            print(f"[{name}] corrected result already checkpointed, skipping", flush=True)
            continue
        df, count_col, path = load_platform(name)
        X = df[EMB].to_numpy(dtype=np.float32)
        y, validation = validate_stored_log1p(df[count_col], path, count_col)
        results.append(run_lasso(X, y, name, count_col, validation))
        save_results(results)
        del df, X, y
        gc.collect()

    print(f"\nSaved {OUTPUT_JSON} (per-platform only)")


if __name__ == "__main__":
    main()
