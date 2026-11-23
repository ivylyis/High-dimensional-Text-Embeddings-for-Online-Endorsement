#!/usr/bin/env python3
"""Leakage-free, platform-specific PCA sensitivity analysis.

The analysis asks how many principal components of the original 384-dimensional
sentence embedding are needed to predict log(1 + endorsement count).  It uses a
fixed, uniformly sampled set of posts per platform and repeated held-out splits.
Within every split, PCA (including its centering) and principal-component
regression are fit on the training observations only.  The raw embedding
coordinates are not variance-standardized: PCA is fit once per split and its
prefix scores are reused for every requested value of k.

Default specification
---------------------
* original 384-dimensional embedding files
* separate Twitter, Yelp, and YouTube analyses
* 10,000 uniformly sampled valid posts per platform
* log(1 + endorsement count) target, read from the already-transformed source field
* 10 deterministic 80/20 train/test splits (seeds 0,...,9)
* raw embeddings centered by train-only PCA; no coordinate-wise variance scaling
* k = 1,2,4,8,16,32,64,128,256,384
* held-out R-squared, summarized with 95% t confidence intervals
* one worker and one BLAS thread

Outputs are checkpointed after every completed split:
  results/pca_sensitivity_splits.csv
  results/pca_sensitivity_summary.csv
  results/pca_sensitivity_results.json
  results/pca_sensitivity_run.log
  results/figures/pca_sensitivity.{pdf,png}

Run with the existing analysis environment:
  python analysis/08_pca_sensitivity.py

Run one platform at a time (useful for low-memory or cloud-backed data):
  python analysis/08_pca_sensitivity.py --platform twitter

The source files may be iCloud placeholders.  This script never copies them and
never writes sampled embeddings or fitted models.  A conservative free-disk
guard runs before a source file is opened.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Cap all common numerical backends before importing NumPy/SciPy/sklearn.
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
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "embeddings"
OUTPUT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "results" / "figures"

EMBEDDING_COLUMNS = [str(i) for i in range(384)]
DEFAULT_K_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384]
DEFAULT_SAMPLE_SIZE = 10_000
DEFAULT_SPLITS = 10
DEFAULT_TEST_SIZE = 0.20
BASE_SAMPLE_SEED = 20_260_818
PLATFORM_SAMPLE_OFFSETS = {"twitter": 11, "yelp": 23, "youtube": 37}

FILES: Dict[str, Tuple[str, str]] = {
    "twitter": ("embedding_endrosment_twitter.parquet", "retweet_count"),
    "yelp": ("embedding_endrosment_yelp.parquet", "useful_count"),
    "youtube": ("embedding_endrosment_youtube.parquet", "likes"),
}

SPLITS_CSV = OUTPUT_DIR / "pca_sensitivity_splits.csv"
SUMMARY_CSV = OUTPUT_DIR / "pca_sensitivity_summary.csv"
RESULTS_JSON = OUTPUT_DIR / "pca_sensitivity_results.json"
RUN_LOG = OUTPUT_DIR / "pca_sensitivity_run.log"
FIGURE_PDF = FIGURE_DIR / "pca_sensitivity.pdf"
FIGURE_PNG = FIGURE_DIR / "pca_sensitivity.png"


def configure_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pca_sensitivity")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(RUN_LOG, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def bytes_gib(value: int) -> float:
    return float(value) / (1024.0 ** 3)


def check_disk_before_open(path: Path, minimum_remaining_gib: float, logger: logging.Logger) -> None:
    """Abort rather than risk exhausting disk while hydrating a cloud file."""
    usage = shutil.disk_usage(ROOT)
    stat = path.stat()
    allocated = int(getattr(stat, "st_blocks", 0)) * 512
    unallocated_logical = max(int(stat.st_size) - allocated, 0)
    projected_remaining = int(usage.free) - unallocated_logical
    logger.info(
        "Disk guard for %s: free=%.2f GiB, logical=%.2f GiB, allocated=%.2f GiB, "
        "worst-case remaining=%.2f GiB",
        path.name,
        bytes_gib(usage.free),
        bytes_gib(stat.st_size),
        bytes_gib(allocated),
        bytes_gib(projected_remaining),
    )
    if projected_remaining < minimum_remaining_gib * (1024.0 ** 3):
        raise RuntimeError(
            f"Disk-safety guard stopped before opening {path.name}: worst-case free "
            f"space after hydration would be {bytes_gib(projected_remaining):.2f} GiB, "
            f"below the {minimum_remaining_gib:.2f} GiB floor."
        )


def select_smallest_priorities(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    sample_keys: np.ndarray,
    batch_x: np.ndarray,
    batch_y: np.ndarray,
    batch_keys: np.ndarray,
    sample_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retain the rows with the smallest random keys (uniform reservoir sample)."""
    if sample_x.size == 0:
        combined_x = batch_x
        combined_y = batch_y
        combined_keys = batch_keys
    else:
        combined_x = np.concatenate((sample_x, batch_x), axis=0)
        combined_y = np.concatenate((sample_y, batch_y), axis=0)
        combined_keys = np.concatenate((sample_keys, batch_keys), axis=0)

    if combined_keys.size <= sample_size:
        return combined_x, combined_y, combined_keys

    keep = np.argpartition(combined_keys, sample_size - 1)[:sample_size]
    return combined_x[keep], combined_y[keep], combined_keys[keep]


def stream_uniform_sample(
    platform: str,
    sample_size: int,
    batch_size: int,
    minimum_remaining_gib: float,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Read only embeddings/outcome and retain a uniform fixed-size sample."""
    file_name, expected_count_column = FILES[platform]
    path = DATA_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(path)
    check_disk_before_open(path, minimum_remaining_gib, logger)

    start = time.time()
    parquet = pq.ParquetFile(path)
    available_columns = list(parquet.schema_arrow.names)
    missing_embeddings = [c for c in EMBEDDING_COLUMNS if c not in available_columns]
    if missing_embeddings:
        raise ValueError(
            f"{path.name} lacks {len(missing_embeddings)} expected embedding columns; "
            f"first missing values: {missing_embeddings[:5]}"
        )

    count_candidates = [
        expected_count_column,
        "retweet_count",
        "useful_count",
        "likes",
        "like_count",
        "useful",
        "endorse_count",
        "endorsement_count",
    ]
    count_column = next((c for c in count_candidates if c in available_columns), None)
    if count_column is None:
        raise ValueError(f"No endorsement-count column found in {path.name}")

    columns = EMBEDDING_COLUMNS + [count_column]
    rng = np.random.RandomState(BASE_SAMPLE_SEED + PLATFORM_SAMPLE_OFFSETS[platform])
    sample_x = np.empty((0, len(EMBEDDING_COLUMNS)), dtype=np.float32)
    sample_y = np.empty(0, dtype=np.float64)
    sample_keys = np.empty(0, dtype=np.float64)
    rows_seen = 0
    valid_rows = 0
    max_count_recovery_error = 0.0
    max_recovered_count = 0.0

    logger.info(
        "[%s] streaming %s rows from %s; keeping a uniform sample of %s",
        platform,
        f"{parquet.metadata.num_rows:,}",
        path.name,
        f"{sample_size:,}",
    )

    for record_batch in parquet.iter_batches(
        batch_size=batch_size, columns=columns, use_threads=False
    ):
        frame = record_batch.to_pandas(use_threads=False)
        batch_x = frame[EMBEDDING_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        stored_log_count = pd.to_numeric(frame[count_column], errors="coerce").to_numpy(
            dtype=np.float64, copy=True
        )
        rows_seen += int(len(frame))
        valid = np.isfinite(stored_log_count) & (stored_log_count >= 0)
        valid &= np.isfinite(batch_x).all(axis=1)
        batch_x = batch_x[valid]
        # The canonical embedding parquets already store log1p(raw count) in
        # their platform-specific outcome fields.  Reapplying log1p here would
        # double-transform the outcome.  Validate this provenance assumption by
        # checking that expm1 recovers nonnegative integer counts (up to floating
        # point error), then use the stored value directly.
        batch_y = stored_log_count[valid]
        recovered_count = np.expm1(batch_y)
        recovery_error = np.abs(recovered_count - np.rint(recovered_count))
        recovery_tolerance = 1e-7 * np.maximum(1.0, np.abs(recovered_count))
        if not np.all(np.isfinite(recovered_count) & (recovery_error <= recovery_tolerance)):
            raise ValueError(
                f"{path.name}:{count_column} does not appear to contain "
                "log1p-transformed nonnegative integer counts"
            )
        if recovered_count.size:
            max_count_recovery_error = max(
                max_count_recovery_error, float(np.max(recovery_error))
            )
            max_recovered_count = max(max_recovered_count, float(np.max(recovered_count)))
        valid_rows += int(batch_y.size)
        if batch_y.size:
            batch_keys = rng.random_sample(batch_y.size)
            sample_x, sample_y, sample_keys = select_smallest_priorities(
                sample_x,
                sample_y,
                sample_keys,
                batch_x,
                batch_y,
                batch_keys,
                sample_size,
            )
        del (
            frame,
            batch_x,
            stored_log_count,
            valid,
            batch_y,
            recovered_count,
            recovery_error,
            recovery_tolerance,
        )

    if sample_y.size < sample_size:
        raise ValueError(
            f"{platform} contains only {sample_y.size:,} valid observations, below the "
            f"requested sample of {sample_size:,}."
        )

    # Canonical order makes downstream splits reproducible even though argpartition
    # does not define the order of retained rows.
    order = np.argsort(sample_keys, kind="mergesort")
    sample_x = np.ascontiguousarray(sample_x[order], dtype=np.float32)
    sample_y = np.ascontiguousarray(sample_y[order], dtype=np.float64)
    elapsed = time.time() - start
    logger.info(
        "[%s] sampled %s of %s valid rows in %.1f s; y mean=%.4f, sd=%.4f",
        platform,
        f"{sample_y.size:,}",
        f"{valid_rows:,}",
        elapsed,
        float(np.mean(sample_y)),
        float(np.std(sample_y, ddof=1)),
    )
    source_info: Dict[str, object] = {
        "path": str(path),
        "logical_bytes": int(path.stat().st_size),
        "parquet_rows": int(parquet.metadata.num_rows),
        "valid_rows": int(valid_rows),
        "rows_seen": int(rows_seen),
        "count_column": count_column,
        "outcome_storage": "source field already equals log1p(raw endorsement count)",
        "outcome_integrity_check": {
            "inverse_transform": "expm1(stored outcome)",
            "expected_values": "nonnegative integer endorsement counts",
            "max_absolute_integer_recovery_error": max_count_recovery_error,
            "maximum_recovered_raw_count": max_recovered_count,
            "passed": True,
        },
        "embedding_dimensions": len(EMBEDDING_COLUMNS),
        "sample_size": int(sample_y.size),
        "sample_seed": int(BASE_SAMPLE_SEED + PLATFORM_SAMPLE_OFFSETS[platform]),
        "sampling_method": "smallest independent random priorities while streaming",
    }
    return sample_x, sample_y, source_info


def orthogonal_prefix_regression(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k_grid: Sequence[int],
) -> Dict[int, float]:
    """OLS on prefixes of orthogonal PC scores, including a fitted intercept."""
    y_mean = float(np.mean(y_train))
    centered_y = y_train - y_mean
    numerator = train_scores.T @ centered_y
    denominator = np.einsum("ij,ij->j", train_scores, train_scores)
    tolerance = (
        float(np.max(denominator))
        * np.finfo(train_scores.dtype).eps
        * max(train_scores.shape)
    )
    coefficients = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > tolerance,
    )
    cumulative_predictions = y_mean + np.cumsum(
        test_scores * coefficients[np.newaxis, :], axis=1
    )
    return {
        int(k): float(r2_score(y_test, cumulative_predictions[:, int(k) - 1]))
        for k in k_grid
    }


def run_one_split(
    x: np.ndarray,
    y: np.ndarray,
    platform: str,
    split_seed: int,
    test_size: float,
    k_grid: Sequence[int],
    logger: logging.Logger,
) -> List[Dict[str, object]]:
    start = time.time()
    indices = np.arange(y.size)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=split_seed, shuffle=True
    )
    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # PCA performs train-mean centering internally.  We deliberately do not
    # variance-standardize embedding coordinates: doing so would change the
    # pretrained embedding geometry and would not match the original notebook.
    # Full SVD is deterministic. It is fit once, then every k uses a prefix.
    pca = PCA(n_components=len(EMBEDDING_COLUMNS), svd_solver="full")
    train_scores = pca.fit_transform(x_train)
    test_scores = pca.transform(x_test)
    r2_by_k = orthogonal_prefix_regression(
        train_scores, test_scores, y_train, y_test, k_grid
    )
    cumulative_explained = np.cumsum(pca.explained_variance_ratio_)
    elapsed = time.time() - start
    logger.info(
        "[%s split=%d] complete in %.1f s; R2(k=1)=%.4f, R2(k=384)=%.4f",
        platform,
        split_seed,
        elapsed,
        r2_by_k[min(k_grid)],
        r2_by_k[max(k_grid)],
    )

    rows: List[Dict[str, object]] = []
    for k in k_grid:
        rows.append(
            {
                "platform": platform,
                "split_seed": int(split_seed),
                "sample_size": int(y.size),
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "test_size": float(test_size),
                "k": int(k),
                "heldout_r2": float(r2_by_k[int(k)]),
                "cumulative_explained_variance": float(cumulative_explained[int(k) - 1]),
                "split_elapsed_seconds": float(elapsed),
            }
        )
    return rows


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_split_results() -> pd.DataFrame:
    if not SPLITS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(SPLITS_CSV)


def checkpoint_rows(existing: pd.DataFrame, new_rows: Iterable[Dict[str, object]]) -> pd.DataFrame:
    new_frame = pd.DataFrame(list(new_rows))
    if existing.empty:
        combined = new_frame
    else:
        keys = set(zip(new_frame["platform"], new_frame["split_seed"], new_frame["k"]))
        retain = [
            (p, int(s), int(k)) not in keys
            for p, s, k in zip(existing["platform"], existing["split_seed"], existing["k"])
        ]
        combined = pd.concat((existing.loc[retain], new_frame), ignore_index=True)
    combined = combined.sort_values(["platform", "split_seed", "k"]).reset_index(drop=True)
    atomic_write_csv(combined, SPLITS_CSV)
    return combined


def summarize_splits(split_results: pd.DataFrame) -> pd.DataFrame:
    summaries: List[Dict[str, object]] = []
    if split_results.empty:
        return pd.DataFrame()
    for (platform, k), group in split_results.groupby(["platform", "k"], sort=True):
        values = group["heldout_r2"].to_numpy(dtype=np.float64)
        explained = group["cumulative_explained_variance"].to_numpy(dtype=np.float64)
        n = int(values.size)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 1 else 0.0
        critical = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else 0.0
        summaries.append(
            {
                "platform": platform,
                "k": int(k),
                "n_splits": n,
                "mean_heldout_r2": mean,
                "sd_heldout_r2": sd,
                "se_heldout_r2": se,
                "ci95_low": mean - critical * se,
                "ci95_high": mean + critical * se,
                "mean_cumulative_explained_variance": float(np.mean(explained)),
                "sd_cumulative_explained_variance": (
                    float(np.std(explained, ddof=1)) if n > 1 else 0.0
                ),
            }
        )
    return pd.DataFrame(summaries).sort_values(["platform", "k"]).reset_index(drop=True)


def classify_curve(platform_summary: pd.DataFrame) -> Dict[str, object]:
    ordered = platform_summary.sort_values("k")
    k_values = ordered["k"].to_numpy(dtype=int)
    r2_values = ordered["mean_heldout_r2"].to_numpy(dtype=float)
    peak_index = int(np.argmax(r2_values))
    peak_r2 = float(r2_values[peak_index])
    peak_k = int(k_values[peak_index])

    def first_fraction(fraction: float) -> int | None:
        if peak_r2 <= 0:
            return None
        candidates = k_values[r2_values >= fraction * peak_r2]
        return int(candidates[0]) if candidates.size else None

    k90 = first_fraction(0.90)
    k95 = first_fraction(0.95)
    if k90 is None:
        pattern = "no_positive_predictive_signal"
    elif k90 <= 16:
        pattern = "low_dimensional_saturation"
    elif k90 <= 64:
        pattern = "moderately_distributed"
    else:
        pattern = "gradual_high_dimensional_gain"

    by_k = dict(zip(k_values.tolist(), r2_values.tolist()))
    return {
        "peak_mean_r2": peak_r2,
        "peak_k": peak_k,
        "mean_r2_at_384": float(by_k.get(384, np.nan)),
        "k_reaching_90pct_of_peak": k90,
        "k_reaching_95pct_of_peak": k95,
        "r2_at_16_as_fraction_of_peak": (
            float(by_k[16] / peak_r2) if peak_r2 > 0 and 16 in by_k else None
        ),
        "curve_pattern": pattern,
    }


def write_summary_outputs(
    split_results: pd.DataFrame,
    source_info: Dict[str, Dict[str, object]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    summary = summarize_splits(split_results)
    if not summary.empty:
        atomic_write_csv(summary, SUMMARY_CSV)

    curve_assessment: Dict[str, Dict[str, object]] = {}
    if not summary.empty:
        for platform, group in summary.groupby("platform"):
            curve_assessment[str(platform)] = classify_curve(group)

    payload: Dict[str, object] = {
        "analysis": "platform-specific principal-component regression sensitivity",
        "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "specification": {
            "embedding_dimensions": 384,
            "sample_size_per_platform": int(args.sample_size),
            "target": (
                "log1p(raw endorsement count), read directly from each canonical "
                "source's already-transformed outcome field"
            ),
            "platform_outcomes": {p: FILES[p][1] for p in FILES},
            "sampling": "fixed uniform random-priority sample from all valid source rows",
            "sample_seed_base": BASE_SAMPLE_SEED,
            "split_seeds": list(range(args.split_seed_start, args.split_seed_start + args.splits)),
            "test_fraction": float(args.test_size),
            "preprocessing": (
                "raw embedding coordinates; no variance standardization; "
                "PCA centers using the training mean"
            ),
            "dimension_reduction": "PCA(full SVD) fit on training observations only",
            "estimator": "ordinary least squares on the first k orthogonal PC scores, with intercept",
            "estimator_rationale": (
                "OLS isolates the predictive information retained by the first k PCs "
                "without adding a second sparsity constraint"
            ),
            "k_grid": [int(k) for k in args.k_grid],
            "metric": "held-out R-squared",
            "uncertainty": "95% t confidence interval over deterministic repeated splits",
            "parallelism": "one worker; common BLAS/OpenMP backends capped at one thread",
        },
        "source_info": source_info,
        "completed_platforms": sorted(split_results["platform"].unique().tolist())
        if not split_results.empty
        else [],
        "completed_split_seeds": {
            str(platform): sorted(group["split_seed"].astype(int).unique().tolist())
            for platform, group in split_results.groupby("platform")
        }
        if not split_results.empty
        else {},
        "curve_assessment": curve_assessment,
        "summary": summary.to_dict(orient="records") if not summary.empty else [],
    }
    temporary = RESULTS_JSON.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    os.replace(temporary, RESULTS_JSON)
    return summary


def plot_summary(summary: pd.DataFrame, logger: logging.Logger) -> None:
    if summary.empty:
        return
    # Do not write Matplotlib caches outside the scoped analysis directory.
    mpl_cache = OUTPUT_DIR / ".mplconfig_pca"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)
    os.environ["XDG_CACHE_HOME"] = str(mpl_cache)
    import matplotlib

    matplotlib.use("Agg")
    # Embed TrueType fonts rather than Type 3 glyphs for journal compatibility.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    styles = {
        "yelp": {"color": "#0072B2", "marker": "o", "linestyle": "-", "label": "Yelp"},
        "twitter": {
            "color": "#D55E00",
            "marker": "s",
            "linestyle": "--",
            "label": "Twitter",
        },
        "youtube": {
            "color": "#009E73",
            "marker": "^",
            "linestyle": "-.",
            "label": "YouTube",
        },
    }
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    all_low: List[float] = []
    all_high: List[float] = []
    for platform in ("yelp", "twitter", "youtube"):
        group = summary.loc[summary["platform"] == platform].sort_values("k")
        if group.empty:
            continue
        k = group["k"].to_numpy(dtype=float)
        x = np.log2(k)
        mean = group["mean_heldout_r2"].to_numpy(dtype=float)
        low = group["ci95_low"].to_numpy(dtype=float)
        high = group["ci95_high"].to_numpy(dtype=float)
        style = styles[platform]
        ax.fill_between(x, low, high, color=style["color"], alpha=0.12, linewidth=0)
        ax.plot(
            x,
            mean,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.1,
            label=style["label"],
        )
        all_low.extend(low.tolist())
        all_high.extend(high.tolist())

    grid = sorted(summary["k"].astype(int).unique().tolist())
    ax.set_xticks(np.log2(np.asarray(grid, dtype=float)))
    ax.set_xticklabels([str(k) for k in grid], rotation=0)
    ax.set_xlabel("Retained principal components ($k$)")
    ax.set_ylabel("Held-out $R^2$")
    ax.axhline(0.0, color="#777777", linewidth=0.7, zorder=0)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if all_low and all_high:
        lower = min(min(all_low) - 0.015, -0.01)
        upper = max(all_high) + 0.025
        ax.set_ylim(lower, upper)
    ax.legend(frameon=False, loc="best", ncol=1)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.xaxis.label.set_size(9.5)
    ax.yaxis.label.set_size(9.5)
    fig.tight_layout(pad=0.5)
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    fig.savefig(FIGURE_PNG, dpi=320, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s and %s", FIGURE_PDF, FIGURE_PNG)


def log_summary(summary: pd.DataFrame, logger: logging.Logger) -> None:
    """Write the same mean/uncertainty values saved to CSV/JSON into the run log."""
    for row in summary.sort_values(["platform", "k"]).itertuples(index=False):
        logger.info(
            "[%s summary] k=%d mean held-out R2=%.6f; 95%% t CI=[%.6f, %.6f]; n_splits=%d",
            row.platform,
            int(row.k),
            float(row.mean_heldout_r2),
            float(row.ci95_low),
            float(row.ci95_high),
            int(row.n_splits),
        )


def synthetic_smoke_test(logger: logging.Logger) -> None:
    """Validate prefix OLS against sklearn LinearRegression without reading data."""
    rng = np.random.RandomState(123)
    x = rng.normal(size=(700, 384)).astype(np.float32)
    y = 0.8 * x[:, 0] - 0.4 * x[:, 5] + rng.normal(scale=0.7, size=700)
    train_idx, test_idx = train_test_split(np.arange(700), test_size=0.2, random_state=0)
    pca = PCA(n_components=384, svd_solver="full").fit(x[train_idx])
    train_scores = pca.transform(x[train_idx])
    test_scores = pca.transform(x[test_idx])
    k_grid = [1, 4, 16, 64, 384]
    fast = orthogonal_prefix_regression(
        train_scores, test_scores, y[train_idx], y[test_idx], k_grid
    )
    for k in k_grid:
        model = LinearRegression().fit(train_scores[:, :k], y[train_idx])
        reference = r2_score(y[test_idx], model.predict(test_scores[:, :k]))
        if not np.isclose(fast[k], reference, rtol=1e-5, atol=1e-6):
            raise AssertionError(
                f"Prefix OLS mismatch at k={k}: fast={fast[k]}, sklearn={reference}"
            )
    logger.info("Synthetic smoke test passed: prefix OLS matches sklearn LinearRegression")


def parse_k_grid(value: str) -> List[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("k grid must contain unique comma-separated integers")
    if min(values) < 1 or max(values) > len(EMBEDDING_COLUMNS):
        raise argparse.ArgumentTypeError("each k must be between 1 and 384")
    return sorted(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=["all", "twitter", "yelp", "youtube"],
        default="all",
        help="platform to run (default: all)",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--splits", type=int, default=DEFAULT_SPLITS)
    parser.add_argument("--split-seed-start", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--k-grid", type=parse_k_grid, default=DEFAULT_K_GRID, help="comma-separated k values"
    )
    parser.add_argument(
        "--minimum-remaining-gib",
        type=float,
        default=0.75,
        help="abort before opening a source if worst-case free disk would fall below this floor",
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="test the numerical pipeline without opening source data or writing results",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_size < 500:
        raise ValueError("sample size must be at least 500")
    if args.splits < 1:
        raise ValueError("splits must be positive")
    if not 0.05 <= args.test_size <= 0.50:
        raise ValueError("test size must lie between 0.05 and 0.50")
    if args.sample_size * (1.0 - args.test_size) <= max(args.k_grid):
        raise ValueError("training sample must contain more observations than the largest k")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    logger = configure_logging()
    logger.info(
        "Thread caps: %s",
        {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"
        )},
    )
    if args.synthetic_smoke:
        synthetic_smoke_test(logger)
        return

    platforms = list(FILES) if args.platform == "all" else [args.platform]
    split_results = load_split_results()
    source_info: Dict[str, Dict[str, object]] = {}
    if RESULTS_JSON.exists():
        try:
            with RESULTS_JSON.open(encoding="utf-8") as handle:
                old_payload = json.load(handle)
            source_info.update(old_payload.get("source_info", {}))
        except (OSError, ValueError, TypeError):
            logger.warning("Could not recover source metadata from %s", RESULTS_JSON)

    requested_seeds = list(range(args.split_seed_start, args.split_seed_start + args.splits))
    for platform in platforms:
        completed: set[int] = set()
        if not split_results.empty:
            platform_rows = split_results.loc[split_results["platform"] == platform]
            expected_k = set(int(k) for k in args.k_grid)
            for seed, group in platform_rows.groupby("split_seed"):
                if set(group["k"].astype(int)) == expected_k:
                    completed.add(int(seed))
        missing_seeds = [seed for seed in requested_seeds if seed not in completed]
        if not missing_seeds:
            logger.info("[%s] all requested splits already checkpointed; skipping", platform)
            continue

        x, y, info = stream_uniform_sample(
            platform,
            args.sample_size,
            args.batch_size,
            args.minimum_remaining_gib,
            logger,
        )
        source_info[platform] = info
        for split_seed in missing_seeds:
            new_rows = run_one_split(
                x,
                y,
                platform,
                split_seed,
                args.test_size,
                args.k_grid,
                logger,
            )
            split_results = checkpoint_rows(split_results, new_rows)
            summary = write_summary_outputs(split_results, source_info, args)
            if not args.no_plot:
                plot_summary(summary, logger)
        del x, y
        logger.info(
            "[%s] platform complete; current free disk %.2f GiB",
            platform,
            bytes_gib(shutil.disk_usage(ROOT).free),
        )

    summary = write_summary_outputs(split_results, source_info, args)
    log_summary(summary, logger)
    if not args.no_plot:
        plot_summary(summary, logger)
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
