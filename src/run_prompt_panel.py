#!/usr/bin/env python3
"""
run_prompt_panel.py

Batch prompt-panel runner for causal attention routing analysis.

Expected prompt CSV columns (minimum):
    prompt_id,prompt_class,prompt_subtype,prompt_text,notes

Optional columns:
    pair_id,token_count_target,word_count_target,entity_count,clause_count,
    has_conditional,has_comparison,has_negation,has_quantifier,
    is_minimal_pair_base,is_minimal_pair_variant

Core outputs:
    - run_manifest.json
    - tokenized_prompt_metadata.csv
    - raw_head_metrics.csv
    - prompt_level_summary.csv
    - optional graph payload JSON files
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from GraphRicciCurvature.FormanRicci import FormanRicci


ALLOWED_CLASSES = {
    "descriptive_baseline",
    "logical_relational",
    "structured_nonlogical",
    "semantic_abstract",
    "minimal_pair",
}


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------

@dataclass
class RunConfig:
    prompt_csv: str
    model_name: str
    layers: list[int] | None
    auto_n_layers: int | None
    edge_threshold: float
    output_dir: str
    max_prompts: int | None
    save_head_curves: bool
    save_graph_payloads: bool
    attn_impl: str
    device: str
    torch_dtype: str
    trust_remote_code: bool


@dataclass
class PromptRecord:
    prompt_id: str
    prompt_class: str
    prompt_subtype: str
    prompt_text: str
    notes: str = ""
    pair_id: str = ""
    token_count_target: str = ""
    word_count_target: str = ""
    entity_count: str = ""
    clause_count: str = ""
    has_conditional: str = ""
    has_comparison: str = ""
    has_negation: str = ""
    has_quantifier: str = ""
    is_minimal_pair_base: str = ""
    is_minimal_pair_variant: str = ""


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Run a prompt panel through a transformer attention-routing analysis."
    )
    parser.add_argument("--prompt_csv", required=True, help="Path to prompt_set_v1.csv")
    parser.add_argument("--model_name", default="gpt2", help="HF model name")

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Explicit layer indices to analyze, e.g. --layers 8 11",
    )
    parser.add_argument(
        "--auto_n_layers",
        type=int,
        default=None,
        help="If set, automatically choose this many evenly spaced layers across model depth",
    )

    parser.add_argument(
        "--edge_threshold",
        type=float,
        default=0.01,
        help="Minimum renormalized causal attention probability to keep an edge",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Optional limit for pilot/debug runs",
    )
    parser.add_argument(
        "--save_head_curves",
        action="store_true",
        help="Reserved flag for future per-prompt plots",
    )
    parser.add_argument(
        "--save_graph_payloads",
        action="store_true",
        help="Save per-prompt/layer/head graph payload JSONs",
    )
    parser.add_argument(
        "--attn_impl",
        default="eager",
        help="Attention implementation passed to AutoModelForCausalLM",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device selection",
    )
    parser.add_argument(
        "--torch_dtype",
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="Torch dtype for model loading",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow remote code for model loading if required",
    )

    args = parser.parse_args()

    if args.layers is None and args.auto_n_layers is None:
        raise ValueError("Provide either --layers or --auto_n_layers")

    if args.layers is not None and args.auto_n_layers is not None:
        raise ValueError("Use either --layers or --auto_n_layers, not both")

    return RunConfig(
        prompt_csv=args.prompt_csv,
        model_name=args.model_name,
        layers=args.layers,
        auto_n_layers=args.auto_n_layers,
        edge_threshold=args.edge_threshold,
        output_dir=args.output_dir,
        max_prompts=args.max_prompts,
        save_head_curves=args.save_head_curves,
        save_graph_payloads=args.save_graph_payloads,
        attn_impl=args.attn_impl,
        device=args.device,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_torch_dtype(dtype_arg: str):
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    return None  # auto


def get_model_depth(model) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    raise ValueError("Could not determine model depth from config")


def choose_evenly_spaced_layers(num_hidden_layers: int, n_select: int) -> list[int]:
    if n_select <= 0:
        raise ValueError("n_select must be positive")
    if n_select == 1:
        return [num_hidden_layers // 2]

    raw = np.linspace(0, num_hidden_layers - 1, n_select)
    layers = sorted(set(int(round(x)) for x in raw))
    return layers


def validate_layers(layers: list[int], num_hidden_layers: int) -> list[int]:
    bad = [layer for layer in layers if layer < 0 or layer >= num_hidden_layers]
    if bad:
        raise ValueError(
            f"Requested invalid layers {bad} for model with {num_hidden_layers} layers"
        )
    return sorted(set(layers))


def collect_model_metadata(model, tokenizer, device: torch.device, dtype_name: str) -> dict[str, Any]:
    cfg = model.config
    return {
        "model_name_or_path": getattr(cfg, "_name_or_path", ""),
        "model_type": getattr(cfg, "model_type", ""),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "device": str(device),
        "torch_dtype": dtype_name,
        "tokenizer_class": tokenizer.__class__.__name__,
    }

def prepare_model_and_tokenizer(config: RunConfig):
    device = resolve_device(config.device)
    torch_dtype = resolve_torch_dtype(config.torch_dtype)

    print(f"--- Initializing model: {config.model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )

    model_kwargs = {
        "attn_implementation": config.attn_impl,
        "trust_remote_code": config.trust_remote_code,
    }

    if torch_dtype is not None:
        model_kwargs["dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs,
    )
    model.eval()
    model.to(device)

    # Optional safety: some tokenizers lack a pad token
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    num_hidden_layers = get_model_depth(model)

    if config.layers is not None:
        layers = validate_layers(config.layers, num_hidden_layers)
    else:
        layers = choose_evenly_spaced_layers(num_hidden_layers, config.auto_n_layers)
        layers = validate_layers(layers, num_hidden_layers)

    model_metadata = collect_model_metadata(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype_name=config.torch_dtype,
    )

    return model, tokenizer, device, layers, model_metadata


# ---------------------------------------------------------------------
# Prompt-table loading / validation
# ---------------------------------------------------------------------

def load_prompt_table(prompt_csv_path: str) -> list[PromptRecord]:
    df = pd.read_csv(prompt_csv_path).fillna("")
    df.columns = df.columns.str.strip()

    core_cols = ["prompt_id", "prompt_class", "prompt_subtype", "prompt_text", "notes"]
    df = df[
        df[core_cols].astype(str).apply(lambda row: any(cell.strip() for cell in row), axis=1)
    ].copy()
    
    required_cols = {"prompt_id", "prompt_class", "prompt_subtype", "prompt_text", "notes"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Prompt CSV missing required columns: {sorted(missing)}")

    rows: list[PromptRecord] = []
    for _, r in df.iterrows():
        row = PromptRecord(
            prompt_id=str(r.get("prompt_id", "")).strip(),
            prompt_class=str(r.get("prompt_class", "")).strip(),
            prompt_subtype=str(r.get("prompt_subtype", "")).strip(),
            prompt_text=str(r.get("prompt_text", "")).strip(),
            notes=str(r.get("notes", "")).strip(),
            pair_id=str(r.get("pair_id", "")).strip(),
            token_count_target=str(r.get("token_count_target", "")).strip(),
            word_count_target=str(r.get("word_count_target", "")).strip(),
            entity_count=str(r.get("entity_count", "")).strip(),
            clause_count=str(r.get("clause_count", "")).strip(),
            has_conditional=str(r.get("has_conditional", "")).strip(),
            has_comparison=str(r.get("has_comparison", "")).strip(),
            has_negation=str(r.get("has_negation", "")).strip(),
            has_quantifier=str(r.get("has_quantifier", "")).strip(),
            is_minimal_pair_base=str(r.get("is_minimal_pair_base", "")).strip(),
            is_minimal_pair_variant=str(r.get("is_minimal_pair_variant", "")).strip(),
        )
        rows.append(row)

    validate_prompt_rows(rows)
    return rows


def validate_prompt_rows(rows: list[PromptRecord]) -> None:
    seen_ids: set[str] = set()

    for row in rows:
        if not row.prompt_id:
            raise ValueError("Found row with empty prompt_id")
        if row.prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt_id found: {row.prompt_id}")
        seen_ids.add(row.prompt_id)

        if not row.prompt_text:
            raise ValueError(f"Prompt {row.prompt_id} has empty prompt_text")

        if row.prompt_class not in ALLOWED_CLASSES:
            raise ValueError(
                f"Prompt {row.prompt_id} has invalid prompt_class '{row.prompt_class}'. "
                f"Allowed: {sorted(ALLOWED_CLASSES)}"
            )
       
        if row.prompt_class == "minimal_pair" and not row.pair_id:
            print(f"[warn] Minimal-pair prompt {row.prompt_id} has no pair_id")

        if row.prompt_class == "minimal_pair":
            def truthy_flag(value: str) -> bool:
                value = str(value).strip().lower()
                return value in {"1", "1.0", "true", "yes", "y"}

            is_base = truthy_flag(row.is_minimal_pair_base)
            is_variant = truthy_flag(row.is_minimal_pair_variant)

            if not (is_base or is_variant):
                print(
                    f"[warn] Minimal-pair prompt {row.prompt_id} is not marked as base or variant"
                )


# ---------------------------------------------------------------------
# Tokenization metadata
# ---------------------------------------------------------------------

def tokenize_prompt(tokenizer, prompt_row: PromptRecord) -> dict[str, Any]:
    inputs = tokenizer(
        prompt_row.prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    token_ids = inputs.input_ids[0].tolist()
    token_strs = tokenizer.convert_ids_to_tokens(token_ids)

    return {
        "prompt_id": prompt_row.prompt_id,
        "prompt_class": prompt_row.prompt_class,
        "prompt_subtype": prompt_row.prompt_subtype,
        "pair_id": prompt_row.pair_id,
        "prompt_text": prompt_row.prompt_text,
        "notes": prompt_row.notes,
        "token_count_actual": len(token_ids),
        "token_ids_json": json.dumps(token_ids),
        "token_strs_json": json.dumps(token_strs),
    }


# ---------------------------------------------------------------------
# Graph metrics
# ---------------------------------------------------------------------

def safe_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def safe_std(values: Iterable[float]) -> float:
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if len(vals) < 2:
        return float("nan")
    return float(np.std(vals, ddof=1))


def compute_row_entropy(attn_h: np.ndarray, seq_len: int) -> tuple[float, float]:
    entropies: list[float] = []

    for i in range(1, seq_len):
        row_probs = attn_h[i, 1:i]
        if len(row_probs) == 0:
            continue

        row_sum = np.sum(row_probs)
        if row_sum <= 0:
            continue

        p = row_probs / row_sum
        p = p[p > 0]
        if len(p) == 0:
            continue

        entropy = -np.sum(p * np.log(p))
        entropies.append(float(entropy))

    return safe_mean(entropies), safe_std(entropies)


def build_causal_graph(attn_h: np.ndarray, seq_len: int, edge_threshold: float) -> nx.DiGraph:
    """
    Mirrors the existing scanner logic:
      - directed causal graph
      - exclude token index 0 from routing backbone
      - row renormalization over 1:i
      - keep edges with p > threshold
      - store transformed distance as edge weight
    """
    G = nx.DiGraph()

    for i in range(1, seq_len):
        row_probs = attn_h[i, 1:i]
        if len(row_probs) == 0:
            continue

        row_sum = np.sum(row_probs)
        if row_sum <= 0:
            continue

        for j_idx, p_raw in enumerate(row_probs):
            p = float(p_raw / row_sum)
            j = j_idx + 1

            if p > edge_threshold:
                w_dist = -np.log(p) + 1e-5
                G.add_edge(j, i, weight=float(w_dist), attn_prob=float(p))

    return G


def compute_graph_metrics(
    G: nx.DiGraph,
    attn_h: np.ndarray,
    seq_len: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """
    Returns:
      metrics_row: dict with scalar metrics
      graph_payload: optional serialized graph payload for later use
    """
    row_entropy_mean, row_entropy_std = compute_row_entropy(attn_h, seq_len)

    if len(G.nodes) == 0 or len(G.edges) == 0:
        metrics = {
            "n_nodes": 0,
            "n_edges": 0,
            "lcc_size": 0,
            "density": float("nan"),
            "mean_forman_curvature": float("nan"),
            "mean_in_degree": float("nan"),
            "mean_out_degree": float("nan"),
            "max_in_degree": float("nan"),
            "max_out_degree": float("nan"),
            "avg_shortest_path_lcc": float("nan"),
            "edge_weight_mean": float("nan"),
            "edge_weight_std": float("nan"),
            "row_entropy_mean": row_entropy_mean,
            "row_entropy_std": row_entropy_std,
        }
        return metrics, None

    lcc_nodes = max(nx.weakly_connected_components(G), key=len)
    G_lcc = G.subgraph(lcc_nodes).copy()

    # Density on LCC
    density = float(nx.density(G_lcc)) if len(G_lcc.nodes) > 1 else float("nan")

    # Degrees
    in_degrees = [deg for _, deg in G_lcc.in_degree()]
    out_degrees = [deg for _, deg in G_lcc.out_degree()]

    # Edge weights
    edge_weights = [float(d["weight"]) for _, _, d in G_lcc.edges(data=True)]

    # Shortest path on underlying undirected version for robustness
    if len(G_lcc.nodes) > 1:
        UG_lcc = G_lcc.to_undirected()
        try:
            avg_shortest_path_lcc = float(nx.average_shortest_path_length(UG_lcc, weight="weight"))
        except Exception:
            avg_shortest_path_lcc = float("nan")
    else:
        avg_shortest_path_lcc = float("nan")

    # Forman-Ricci
    try:
        frc = FormanRicci(G_lcc)
        frc.compute_ricci_curvature()
        ks = [d["formanCurvature"] for _, _, d in frc.G.edges(data=True)]
        mean_forman_curvature = safe_mean(ks)
    except Exception as e:
        print(f"[warn] Forman-Ricci failed: {e}")
        mean_forman_curvature = float("nan")

    metrics = {
        "n_nodes": int(len(G_lcc.nodes)),
        "n_edges": int(len(G_lcc.edges)),
        "lcc_size": int(len(G_lcc.nodes)),
        "density": density,
        "mean_forman_curvature": mean_forman_curvature,
        "mean_in_degree": safe_mean(in_degrees),
        "mean_out_degree": safe_mean(out_degrees),
        "max_in_degree": float(max(in_degrees)) if in_degrees else float("nan"),
        "max_out_degree": float(max(out_degrees)) if out_degrees else float("nan"),
        "avg_shortest_path_lcc": avg_shortest_path_lcc,
        "edge_weight_mean": safe_mean(edge_weights),
        "edge_weight_std": safe_std(edge_weights),
        "row_entropy_mean": row_entropy_mean,
        "row_entropy_std": row_entropy_std,
    }

    graph_payload = {
        "nodes": list(map(int, G_lcc.nodes())),
        "edges": [
            {
                "source": int(u),
                "target": int(v),
                "weight": float(d["weight"]),
                "attn_prob": float(d.get("attn_prob", float("nan"))),
            }
            for u, v, d in G_lcc.edges(data=True)
        ],
    }

    return metrics, graph_payload


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

def run_prompt_panel(
    model,
    tokenizer,
    device: torch.device,
    prompt_rows: list[PromptRecord],
    layers: list[int],
    edge_threshold: float,
    save_graph_payloads: bool,
    graph_payload_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns:
      token_metadata_rows
      raw_head_metric_rows
      graph_payload_manifest_rows
    """
    token_metadata_rows: list[dict[str, Any]] = []
    raw_head_metric_rows: list[dict[str, Any]] = []
    graph_payload_manifest_rows: list[dict[str, Any]] = []

    for idx, prompt_row in enumerate(prompt_rows, start=1):
        print(f"[{idx}/{len(prompt_rows)}] Processing {prompt_row.prompt_id} ({prompt_row.prompt_class})")

        token_meta = tokenize_prompt(tokenizer, prompt_row)
        token_metadata_rows.append(token_meta)

        inputs = tokenizer(
            prompt_row.prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attentions = outputs.attentions
        actual_token_count = token_meta["token_count_actual"]

        for layer_idx in layers:
            layer_attn = attentions[layer_idx][0].cpu().numpy()  # shape: [n_heads, seq_len, seq_len]
            n_heads, seq_len, _ = layer_attn.shape

            for head_idx in range(n_heads):
                attn_h = layer_attn[head_idx]
                G = build_causal_graph(attn_h, seq_len, edge_threshold)
                metrics, graph_payload = compute_graph_metrics(G, attn_h, seq_len)

                row = {
                    "prompt_id": prompt_row.prompt_id,
                    "prompt_class": prompt_row.prompt_class,
                    "prompt_subtype": prompt_row.prompt_subtype,
                    "pair_id": prompt_row.pair_id,
                    "prompt_text": prompt_row.prompt_text,
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "token_count_actual": int(actual_token_count),
                    **metrics,
                }
                raw_head_metric_rows.append(row)

                if save_graph_payloads and graph_payload is not None:
                    payload_filename = f"{prompt_row.prompt_id}_layer{layer_idx}_head{head_idx}.json"
                    payload_path = graph_payload_dir / payload_filename
                    with payload_path.open("w", encoding="utf-8") as f:
                        json.dump(graph_payload, f, indent=2)

                    graph_payload_manifest_rows.append(
                        {
                            "prompt_id": prompt_row.prompt_id,
                            "prompt_class": prompt_row.prompt_class,
                            "prompt_subtype": prompt_row.prompt_subtype,
                            "pair_id": prompt_row.pair_id,
                            "prompt_text": prompt_row.prompt_text,
                            "layer": int(layer_idx),
                            "head": int(head_idx),
                            "payload_path": str(payload_path),
                        }
                    )

    return token_metadata_rows, raw_head_metric_rows, graph_payload_manifest_rows


# ---------------------------------------------------------------------
# Summaries / writeout
# ---------------------------------------------------------------------

def summarize_prompt_level(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "prompt_id",
        "prompt_class",
        "prompt_subtype",
        "pair_id",
        "prompt_text",
        "layer",
        "token_count_actual",
    ]

    metric_cols = [
        "mean_forman_curvature",
        "density",
        "n_edges",
        "lcc_size",
        "row_entropy_mean",
        "avg_shortest_path_lcc",
    ]

    agg_spec: dict[str, list[str]] = {col: ["mean", "std"] for col in metric_cols}

    summary = raw_df.groupby(group_cols, dropna=False).agg(agg_spec)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()

    return summary


def write_run_manifest(
    config: RunConfig,
    output_dir: Path,
    n_prompts: int,
    model_metadata: dict[str, Any],
    resolved_layers: list[int],
) -> None:
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "n_prompts": n_prompts,
        "resolved_layers": resolved_layers,
        "model_metadata": model_metadata,
    }

    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main() -> None:
    config = parse_args()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_payload_dir = output_dir / "graph_payloads"
    if config.save_graph_payloads:
        graph_payload_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompt_rows = load_prompt_table(config.prompt_csv)
    if config.max_prompts is not None:
        prompt_rows = prompt_rows[: config.max_prompts]

    model, tokenizer, device, resolved_layers, model_metadata = prepare_model_and_tokenizer(config)

    write_run_manifest(
        config=config,
        output_dir=output_dir,
        n_prompts=len(prompt_rows),
        model_metadata=model_metadata,
        resolved_layers=resolved_layers,
    )

    token_metadata_rows, raw_head_metric_rows, graph_payload_manifest_rows = run_prompt_panel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt_rows=prompt_rows,
        layers=resolved_layers,
        edge_threshold=config.edge_threshold,
        save_graph_payloads=config.save_graph_payloads,
        graph_payload_dir=graph_payload_dir,
    )

    # Write token metadata
    token_meta_df = pd.DataFrame(token_metadata_rows)
    token_meta_path = output_dir / "tokenized_prompt_metadata.csv"
    token_meta_df.to_csv(token_meta_path, index=False)

    # Write raw metrics
    raw_df = pd.DataFrame(raw_head_metric_rows)
    raw_path = output_dir / "raw_head_metrics.csv"
    raw_df.to_csv(raw_path, index=False)

    # Write prompt summaries
    summary_df = summarize_prompt_level(raw_df)
    summary_path = output_dir / "prompt_level_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Optional payload manifest
    if config.save_graph_payloads:
        payload_manifest_df = pd.DataFrame(graph_payload_manifest_rows)
        payload_manifest_df.to_csv(output_dir / "graph_payload_manifest.csv", index=False)

    print("\n--- Done ---")
    print(f"Saved token metadata to: {token_meta_path}")
    print(f"Saved raw metrics to:    {raw_path}")
    print(f"Saved prompt summary to: {summary_path}")


if __name__ == "__main__":
    main()