#!/usr/bin/env python3
"""
plot_results.py

Generate summary plots from:
    - class_layer_summary.csv
    - class_layer_head_summary.csv
    - optional minimal_pair_differences.csv
    - optional graph_payload_manifest.csv

Outputs:
    - fig1_class_layer_overview.png
    - fig2_logical_minus_descriptive_heatmap.png
    - fig3_logical_minus_structured_heatmap.png
    - fig4_minimal_pair_effects.png
    - fig5_topology_exemplars.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class PlotConfig:
    class_layer_csv: str
    class_layer_head_csv: str
    output_dir: str
    minimal_pair_csv: str | None
    graph_payload_manifest_csv: str | None
    token_metadata_csv: str | None
    model_label: str

CLASS_ORDER = [
    "semantic_abstract",
    "structured_nonlogical",
    "descriptive_baseline",
    "minimal_pair",
    "logical_relational",
]

CLASS_LABELS = {
    "semantic_abstract": "Semantic Abstract",
    "structured_nonlogical": "Structured Nonlogical",
    "descriptive_baseline": "Descriptive Baseline",
    "minimal_pair": "Minimal Pair",
    "logical_relational": "Logical Relational",
}

METRICS_FOR_FIG1 = [
    ("mean_forman_curvature_mean", "mean_forman_curvature_ci_lo", "mean_forman_curvature_ci_hi", "Mean Forman Curvature"),
    ("density_mean", "density_ci_lo", "density_ci_hi", "Density"),
    ("row_entropy_mean_mean", "row_entropy_mean_ci_lo", "row_entropy_mean_ci_hi", "Row Entropy"),
    ("avg_shortest_path_lcc_mean", "avg_shortest_path_lcc_ci_lo", "avg_shortest_path_lcc_ci_hi", "Avg Shortest Path"),
]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> PlotConfig:
    parser = argparse.ArgumentParser(description="Plot summary results for LLM routing study.")
    parser.add_argument("--class_layer_csv", required=True, help="Path to class_layer_summary.csv")
    parser.add_argument("--class_layer_head_csv", required=True, help="Path to class_layer_head_summary.csv")
    parser.add_argument("--output_dir", required=True, help="Directory to save figures")
    parser.add_argument("--minimal_pair_csv", default=None, help="Optional path to minimal_pair_differences.csv")
    parser.add_argument("--graph_payload_manifest_csv", default=None, help="Optional path to graph_payload_manifest.csv")
    parser.add_argument(
        "--token_metadata_csv",
        default=None,
        help="Optional path to tokenized_prompt_metadata.csv for token labels in topology exemplars",
    )
    parser.add_argument("--model_label", default="", help="Optional model label for figure titles")
    args = parser.parse_args()

    return PlotConfig(
        class_layer_csv=args.class_layer_csv,
        class_layer_head_csv=args.class_layer_head_csv,
        output_dir=args.output_dir,
        minimal_pair_csv=args.minimal_pair_csv,
        graph_payload_manifest_csv=args.graph_payload_manifest_csv,
        token_metadata_csv=args.token_metadata_csv,
        model_label=args.model_label,
    )


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def load_token_metadata(token_metadata_csv: str) -> dict[str, list[str]]:
    """
    Returns:
        dict mapping prompt_id -> list of token strings
    """
    df = load_csv(token_metadata_csv)

    required_cols = {"prompt_id", "token_strs_json"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"tokenized_prompt_metadata.csv missing columns: {sorted(missing)}")

    token_map: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        prompt_id = str(row["prompt_id"]).strip()
        try:
            token_strs = json.loads(row["token_strs_json"])
        except Exception:
            token_strs = []
        token_map[prompt_id] = token_strs

    return token_map


# ---------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------

def plot_class_layer_overview(
    class_layer_df: pd.DataFrame,
    output_dir: Path,
    model_label: str = "",
) -> None:

    layers = sorted(class_layer_df["layer"].dropna().unique().tolist())

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for ax, (mean_col, ci_lo_col, ci_hi_col, title) in zip(axs, METRICS_FOR_FIG1):
        for layer in layers:
            subset = class_layer_df[class_layer_df["layer"] == layer].copy()
            subset["prompt_class"] = pd.Categorical(
                subset["prompt_class"], categories=CLASS_ORDER, ordered=True
            )
            subset = subset.sort_values("prompt_class")

            x = np.arange(len(subset))
            y = subset[mean_col].to_numpy(dtype=float)

            if ci_lo_col in subset.columns and ci_hi_col in subset.columns:
                lo = subset[ci_lo_col].to_numpy(dtype=float)
                hi = subset[ci_hi_col].to_numpy(dtype=float)
                yerr = np.vstack([y - lo, hi - y])
            else:
                yerr = None

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linestyle="-",
                capsize=4,
                label=f"Layer {layer}",
            )

        ax.set_title(title)
        ax.set_xticks(np.arange(len(CLASS_ORDER)))
        ax.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER], rotation=25, ha="right")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()

    title_prefix = f"{model_label} - " if model_label else ""
    fig.suptitle(f"{title_prefix}Class-by-Layer Summary Overview", fontsize=16)
    
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = output_dir / "fig1_class_layer_overview.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------

def build_delta_matrix(
    class_layer_head_df: pd.DataFrame,
    class_a: str,
    class_b: str,
    metric_col: str = "mean_forman_curvature_mean",
) -> tuple[np.ndarray, list[int], list[int]]:
    layers = sorted(class_layer_head_df["layer"].dropna().unique().tolist())
    heads = sorted(class_layer_head_df["head"].dropna().unique().tolist())

    matrix = np.full((len(layers), len(heads)), np.nan)

    for i, layer in enumerate(layers):
        for j, head in enumerate(heads):
            a = class_layer_head_df[
                (class_layer_head_df["prompt_class"] == class_a)
                & (class_layer_head_df["layer"] == layer)
                & (class_layer_head_df["head"] == head)
            ]
            b = class_layer_head_df[
                (class_layer_head_df["prompt_class"] == class_b)
                & (class_layer_head_df["layer"] == layer)
                & (class_layer_head_df["head"] == head)
            ]

            if len(a) == 1 and len(b) == 1:
                a_val = float(a.iloc[0][metric_col])
                b_val = float(b.iloc[0][metric_col])
                matrix[i, j] = a_val - b_val

    return matrix, layers, heads


def plot_heatmap(
    matrix: np.ndarray,
    layers: list[int],
    heads: list[int],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))

    max_abs = np.nanmax(np.abs(matrix))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )

    ax.set_title(title)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(np.arange(len(heads)))
    ax.set_xticklabels(heads)
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels(layers)

    for i in range(len(layers)):
        for j in range(len(heads)):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta in mean Forman curvature")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------
# Figure 4: Minimal-pair effects
# ---------------------------------------------------------------------

def plot_minimal_pair_effects(
    minimal_pair_df: pd.DataFrame,
    output_dir: Path,
    model_label: str = "",
) -> None:

    """
    Plot minimal-pair differences by layer.
    Uses *_diff columns from minimal_pair_differences.csv.
    """
    if minimal_pair_df.empty:
        print("[warn] minimal_pair_df is empty; skipping Figure 4")
        return

    metrics = [
        ("mean_forman_curvature_diff", "Mean Forman Curvature Diff"),
        ("density_diff", "Density Diff"),
        ("row_entropy_mean_diff", "Row Entropy Diff"),
    ]

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    axs = np.atleast_1d(axs)

    pair_ids = sorted(minimal_pair_df["pair_id"].dropna().unique().tolist())

    for ax, (metric_col, title) in zip(axs, metrics):
        for pair_id in pair_ids:
            subset = minimal_pair_df[minimal_pair_df["pair_id"] == pair_id].copy()
            subset = subset.sort_values("layer")

            if metric_col not in subset.columns:
                continue

            ax.plot(
                subset["layer"].to_numpy(),
                subset[metric_col].to_numpy(dtype=float),
                marker="o",
                linestyle="-",
                alpha=0.7,
                label=pair_id,
            )

        # bold mean line
        mean_by_layer = (
            minimal_pair_df.groupby("layer", dropna=False)[metric_col]
            .mean()
            .reset_index()
            .sort_values("layer")
        )
        ax.plot(
            mean_by_layer["layer"].to_numpy(),
            mean_by_layer[metric_col].to_numpy(dtype=float),
            marker="o",
            linestyle="-",
            linewidth=3,
            color="black",
            label="Mean",
        )

        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.grid(True, linestyle=":", alpha=0.5)

    axs[0].set_ylabel("Logical variant - Base prompt")
    axs[-1].legend(loc="best", fontsize=8)

    title_prefix = f"{model_label} - " if model_label else ""
    fig.suptitle(f"{title_prefix}Minimal-Pair Effects Across Layers", fontsize=16)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = output_dir / "fig4_minimal_pair_effects.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------
# Figure 5: Topology exemplars
# ---------------------------------------------------------------------

def clean_token_label(token: str) -> str:
    token = str(token)
    token = token.replace("Ġ", " ")
    token = token.replace("Ċ", "\\n")
    token = token.strip()

    if token == "":
        token = "[space]"

    return token

def build_node_label_map(prompt_id: str, nodes: list[int], token_map: dict[str, list[str]] | None) -> dict[int, str]:
    """
    Map graph node indices to readable token labels.
    """
    if token_map is None or prompt_id not in token_map:
        return {int(node): str(node) for node in nodes}

    token_strs = token_map[prompt_id]
    labels: dict[int, str] = {}

    for node in nodes:
        node = int(node)
        if 0 <= node < len(token_strs):
            labels[node] = clean_token_label(token_strs[node])
        else:
            labels[node] = str(node)

    return labels

def load_graph_payload(payload_path: Path) -> dict:
    with payload_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def payload_to_graph(payload: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in payload.get("nodes", []):
        G.add_node(int(node))
    for edge in payload.get("edges", []):
        G.add_edge(
            int(edge["source"]),
            int(edge["target"]),
            weight=float(edge.get("weight", 1.0)),
            attn_prob=float(edge.get("attn_prob", np.nan)),
        )
    return G


def choose_exemplar_rows(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Choose three exemplars:
      - descriptive_baseline
      - structured_nonlogical
      - logical_relational

    Qwen-oriented preference:
      deeper layers first, since the strongest class separation appeared
      later in depth than in GPT-2.
    """
    preferred_classes = [
        "descriptive_baseline",
        "structured_nonlogical",
        "logical_relational",
    ]

    preferred_layers = [15, 23, 8, 0]
    preferred_heads = [11, 6, 2, 7, 0, 3, 9, 12, 1, 4, 5, 8, 10, 13]

    rows = []

    for prompt_class in preferred_classes:
        subset = manifest_df[manifest_df["prompt_class"] == prompt_class].copy()
        if subset.empty:
            continue

        subset["layer_rank"] = subset["layer"].apply(
            lambda x: preferred_layers.index(x) if x in preferred_layers else 999
        )
        subset["head_rank"] = subset["head"].apply(
            lambda x: preferred_heads.index(x) if x in preferred_heads else 999
        )
        subset = subset.sort_values(["layer_rank", "head_rank", "prompt_id"])

        rows.append(subset.iloc[0])

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def draw_payload_graph(
    ax,
    payload: dict,
    title: str,
    prompt_id: str,
    token_map: dict[str, list[str]] | None = None,
) -> None:

    G = payload_to_graph(payload)

    if len(G.nodes) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "Empty graph", ha="center", va="center")
        ax.axis("off")
        return

    # use undirected for stable layout
    UG = G.to_undirected()
    try:
        pos = nx.kamada_kawai_layout(UG, weight="weight")
    except Exception:
        pos = nx.spring_layout(UG, seed=1)

    edge_weights = [d.get("attn_prob", 0.1) for _, _, d in G.edges(data=True)]
    if len(edge_weights) == 0:
        widths = 1.0
    else:
        ew = np.array(edge_weights, dtype=float)
        widths = 1.0 + 6.0 * (ew / np.nanmax(ew))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600)
    node_labels = build_node_label_map(prompt_id, list(G.nodes()), token_map)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=7)

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=widths,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
        alpha=0.7,
    )

    ax.set_title(title)
    ax.axis("off")


def plot_topology_exemplars(
    graph_payload_manifest_df: pd.DataFrame,
    output_dir: Path,
    token_map: dict[str, list[str]] | None = None,
    model_label: str = "",
) -> None:

    if graph_payload_manifest_df.empty:
        print("[warn] graph_payload_manifest_df is empty; skipping Figure 5")
        return

    exemplar_rows = choose_exemplar_rows(graph_payload_manifest_df)
    if exemplar_rows.empty:
        print("[warn] No exemplar rows found; skipping Figure 5")
        return

    n = len(exemplar_rows)
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 5))
    axs = np.atleast_1d(axs)

    for ax, (_, row) in zip(axs, exemplar_rows.iterrows()):
        payload_path = Path(row["payload_path"])
        payload = load_graph_payload(payload_path)

        title = (
            f"{row['prompt_class']}\n"
            f"{row['prompt_id']} | layer {row['layer']} | head {row['head']}"
        )
        draw_payload_graph(
            ax,
            payload,
            title,
            prompt_id=row["prompt_id"],
            token_map=token_map,
        )

    title_prefix = f"{model_label} - " if model_label else ""
    fig.suptitle(f"{title_prefix}Representative Topology Exemplars", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = output_dir / "fig5_topology_exemplars.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_layer_df = load_csv(config.class_layer_csv)
    class_layer_head_df = load_csv(config.class_layer_head_csv)

    token_map = None
    if config.token_metadata_csv is not None:
        token_map = load_token_metadata(config.token_metadata_csv)

    # Figure 1
    plot_class_layer_overview(class_layer_df, output_dir, model_label=config.model_label)

    # Figure 2: logical - descriptive
    matrix_ld, layers_ld, heads_ld = build_delta_matrix(
        class_layer_head_df,
        class_a="logical_relational",
        class_b="descriptive_baseline",
        metric_col="mean_forman_curvature_mean",
    )
    plot_heatmap(
        matrix_ld,
        layers_ld,
        heads_ld,
        title="Logical Relational minus Descriptive Baseline\n(Mean Forman Curvature)",
        output_path=output_dir / "fig2_logical_minus_descriptive_heatmap.png",
    )

    # Figure 3: logical - structured nonlogical
    matrix_ls, layers_ls, heads_ls = build_delta_matrix(
        class_layer_head_df,
        class_a="logical_relational",
        class_b="structured_nonlogical",
        metric_col="mean_forman_curvature_mean",
    )
    plot_heatmap(
        matrix_ls,
        layers_ls,
        heads_ls,
        title="Logical Relational minus Structured Nonlogical\n(Mean Forman Curvature)",
        output_path=output_dir / "fig3_logical_minus_structured_heatmap.png",
    )

    # Figure 4: minimal-pair effects
    if config.minimal_pair_csv is not None:
        minimal_pair_df = load_csv(config.minimal_pair_csv)
        plot_minimal_pair_effects(minimal_pair_df, output_dir, model_label=config.model_label)
    else:
        print("[info] No --minimal_pair_csv provided; skipping Figure 4")

    # Figure 5: topology exemplars
    if config.graph_payload_manifest_csv is not None:
        graph_payload_manifest_df = load_csv(config.graph_payload_manifest_csv)
        plot_topology_exemplars(
            graph_payload_manifest_df,
            output_dir,
            token_map=token_map,
            model_label=config.model_label,
        )

    else:
        print("[info] No --graph_payload_manifest_csv provided; skipping Figure 5")


if __name__ == "__main__":
    main()