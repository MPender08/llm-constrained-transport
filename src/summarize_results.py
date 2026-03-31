#!/usr/bin/env python3
"""
summarize_results.py

Aggregate and summarize prompt-panel results from run_prompt_panel.py.

Expected input:
    raw_head_metrics.csv

Primary outputs:
    - class_layer_summary.csv
    - class_layer_head_summary.csv
    - minimal_pair_differences.csv
    - summary_manifest.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class SummaryConfig:
    input_csv: str
    output_dir: str


SUMMARY_METRICS = [
    "mean_forman_curvature",
    "density",
    "n_edges",
    "lcc_size",
    "row_entropy_mean",
    "avg_shortest_path_lcc",
]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> SummaryConfig:
    parser = argparse.ArgumentParser(
        description="Summarize raw transformer routing metrics across prompt classes and minimal pairs."
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to raw_head_metrics.csv produced by run_prompt_panel.py",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write summary tables into",
    )

    args = parser.parse_args()
    return SummaryConfig(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
    )


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return float("nan")
    return float(vals.mean())


def safe_std(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) < 2:
        return float("nan")
    return float(vals.std(ddof=1))


def safe_median(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return float("nan")
    return float(vals.median())


def bootstrap_ci(series: pd.Series, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])

    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means.append(sample.mean())

    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    df.columns = [
        "_".join([str(part) for part in col if str(part) != ""]).strip("_")
        for col in df.columns
    ]
    return df


# ---------------------------------------------------------------------
# Load / validate
# ---------------------------------------------------------------------

def load_raw_metrics(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()

    required_cols = {
        "prompt_id",
        "prompt_class",
        "prompt_subtype",
        "pair_id",
        "prompt_text",
        "layer",
        "head",
    } | set(SUMMARY_METRICS)

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    return df


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def summarize_class_layer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across all rows within prompt_class x layer.
    This mixes heads and prompts together, which is acceptable for a high-level first pass.
    """
    group_cols = ["prompt_class", "layer"]

    agg_dict = {}
    for metric in SUMMARY_METRICS:
        agg_dict[metric] = ["mean", "std", "median"]

    summary = df.groupby(group_cols, dropna=False).agg(agg_dict)
    summary = flatten_multiindex_columns(summary).reset_index()

    # Add bootstrap CIs for metric means
    ci_rows = []
    for (prompt_class, layer), g in df.groupby(group_cols, dropna=False):
        row = {"prompt_class": prompt_class, "layer": layer}
        for metric in SUMMARY_METRICS:
            lo, hi = bootstrap_ci(g[metric])
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi
        ci_rows.append(row)

    ci_df = pd.DataFrame(ci_rows)
    summary = summary.merge(ci_df, on=["prompt_class", "layer"], how="left")
    return summary


def summarize_class_layer_head(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate within prompt_class x layer x head.
    Useful for head-localization heatmaps later.
    """
    group_cols = ["prompt_class", "layer", "head"]

    agg_dict = {}
    for metric in SUMMARY_METRICS:
        agg_dict[metric] = ["mean", "std", "median"]

    summary = df.groupby(group_cols, dropna=False).agg(agg_dict)
    summary = flatten_multiindex_columns(summary).reset_index()

    ci_rows = []
    for (prompt_class, layer, head), g in df.groupby(group_cols, dropna=False):
        row = {"prompt_class": prompt_class, "layer": layer, "head": head}
        for metric in SUMMARY_METRICS:
            lo, hi = bootstrap_ci(g[metric])
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi
        ci_rows.append(row)

    ci_df = pd.DataFrame(ci_rows)
    summary = summary.merge(ci_df, on=["prompt_class", "layer", "head"], how="left")
    return summary


def summarize_prompt_layer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse across heads first, yielding one row per prompt_id x layer.
    This is useful for cleaner minimal-pair comparisons.
    """
    group_cols = [
        "prompt_id",
        "prompt_class",
        "prompt_subtype",
        "pair_id",
        "prompt_text",
        "layer",
    ]

    agg_dict = {}
    for metric in SUMMARY_METRICS:
        agg_dict[metric] = ["mean", "std"]

    summary = df.groupby(group_cols, dropna=False).agg(agg_dict)
    summary = flatten_multiindex_columns(summary).reset_index()
    return summary


def compute_minimal_pair_differences(prompt_layer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise differences within minimal_pair prompts.

    Requires:
        - pair_id populated
        - prompt_subtype distinguishing base vs logical variant

    Convention used:
        diff = logical_pair_variant - descriptive_pair_base
    """
    mp = prompt_layer_df[prompt_layer_df["prompt_class"] == "minimal_pair"].copy()

    if mp.empty:
        return pd.DataFrame()

    if "pair_id" not in mp.columns:
        return pd.DataFrame()

    mp = mp[mp["pair_id"].astype(str).str.strip() != ""].copy()
    if mp.empty:
        return pd.DataFrame()

    rows = []

    for (pair_id, layer), g in mp.groupby(["pair_id", "layer"], dropna=False):
        base = g[g["prompt_subtype"] == "descriptive_pair_base"]
        variant = g[g["prompt_subtype"] == "logical_pair_variant"]

        if len(base) != 1 or len(variant) != 1:
            # Skip malformed or incomplete pair groups
            continue

        base_row = base.iloc[0]
        var_row = variant.iloc[0]

        out = {
            "pair_id": pair_id,
            "layer": layer,
            "base_prompt_id": base_row["prompt_id"],
            "variant_prompt_id": var_row["prompt_id"],
            "base_prompt_text": base_row["prompt_text"],
            "variant_prompt_text": var_row["prompt_text"],
        }

        for metric in SUMMARY_METRICS:
            base_val = base_row.get(f"{metric}_mean", np.nan)
            var_val = var_row.get(f"{metric}_mean", np.nan)
            out[f"{metric}_base"] = base_val
            out[f"{metric}_variant"] = var_val
            out[f"{metric}_diff"] = var_val - base_val

        rows.append(out)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------

def write_summary_manifest(config: SummaryConfig, output_dir: Path, input_rows: int) -> None:
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "input_rows": input_rows,
        "summary_metrics": SUMMARY_METRICS,
    }
    with (output_dir / "summary_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    config = parse_args()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_metrics(config.input_csv)

    class_layer_df = summarize_class_layer(df)
    class_layer_head_df = summarize_class_layer_head(df)
    prompt_layer_df = summarize_prompt_layer(df)
    minimal_pair_df = compute_minimal_pair_differences(prompt_layer_df)

    class_layer_path = output_dir / "class_layer_summary.csv"
    class_layer_head_path = output_dir / "class_layer_head_summary.csv"
    prompt_layer_path = output_dir / "prompt_layer_summary.csv"
    minimal_pair_path = output_dir / "minimal_pair_differences.csv"

    class_layer_df.to_csv(class_layer_path, index=False)
    class_layer_head_df.to_csv(class_layer_head_path, index=False)
    prompt_layer_df.to_csv(prompt_layer_path, index=False)
    minimal_pair_df.to_csv(minimal_pair_path, index=False)

    write_summary_manifest(config, output_dir, input_rows=len(df))

    print("\n--- Summary complete ---")
    print(f"Saved class-layer summary to:      {class_layer_path}")
    print(f"Saved class-layer-head summary to: {class_layer_head_path}")
    print(f"Saved prompt-layer summary to:     {prompt_layer_path}")
    print(f"Saved minimal-pair diffs to:       {minimal_pair_path}")


if __name__ == "__main__":
    main()