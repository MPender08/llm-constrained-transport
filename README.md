***

# Formal Constraint and Routing Reorganization: A Constrained-Transport View of Transformer Attention

**Author:** Matthew A. Pender

**Preprint / Zenodo DOI:** [https://doi.org/10.5281/zenodo.19363505](https://doi.org/10.5281/zenodo.19363505)

The project investigates whether logical and relational prompts induce a distinct internal routing regime in transformer attention, relative to descriptive baseline, structured nonlogical, semantic-abstract, and minimal-pair controls. The current workflow supports cross-model analysis in **GPT-2** and **Qwen 0.5B**, with a shared graph-construction and summary pipeline.

![fig1_class_layer_overview](results/qwen_0p5b/figures/fig1_class_layer_overview.png)

**Figure 6: Class-level routing structure in Qwen 0.5B across prompt families.** Class-by-layer summary of internal routing descriptors for Qwen 0.5B across the five prompt classes at layers 0, 8, 15, and 23. Logical/relational prompts again occupy the highest regime on the principal routing descriptors, while semantic-abstract prompts remain comparatively low and structured nonlogical prompts generally fall below descriptive baseline. The strongest class separation appears in the deeper layers, especially layer 15, indicating that the logical-routing effect survives transfer to a second model family but is expressed later in depth than in GPT-2. Error bars indicate bootstrap confidence intervals over prompt × head observations.

## Overview

The central idea of this project is to treat **causal attention as a transport graph**.

For each prompt, the pipeline:

1. extracts causal attention tensors from selected transformer layers,
2. converts each attention head into a directed routing graph,
3. computes graph-level descriptors such as:
   - mean Forman curvature,
   - density,
   - row entropy,
   - average shortest path,
4. aggregates results across prompts, classes, layers, and heads,
5. generates summary tables and figures for manuscript-ready analysis.

### Prompt Classes

The current prompt panel includes five classes:

* **descriptive_baseline:** Ordinary scene-level or event-level prompts without explicit deductive structure.
* **logical_relational:** Prompts emphasizing comparison, relational dependence, or explicit inferential form.
* **structured_nonlogical:** Prompts that are organized and compositionally tight but not explicitly inferential.
* **semantic_abstract:** Prompts containing conceptual or evocative content without explicit logical scaffolding.
* **minimal_pair:** Closely matched descriptive/logical prompt pairs used for tighter local comparisons.

### Core Method

For each prompt and selected layer:

* extract output_attentions=True
* analyze each head separately
* build a directed causal graph using:
    * causal masking,
    * exclusion of token index 0 from the routing backbone,
    * row renormalization over admissible causal history,
    *edge retention above a specified threshold
* define edge distance as:
```
w = -log(p) + ε
```
where p is the renormalized causal attention probability.

Metrics are computed on the largest weakly connected component.

### Summary Metrics

The current analysis tracks:

* mean_forman_curvature
* density
* n_edges
* lcc_size
* row_entropy_mean
* avg_shortest_path_lcc

These are saved at the raw head level and then summarized across prompt classes, layers, heads, and minimal-pair conditions.

***

## Repository Structure

The repository is organized as follows:

```text
llm_constrained_tranpsort/
│
├── src/     
│   ├── run_prompt_panel.py    #  Runs the prompt panel, extracts attention graphs, computes per-head graph metrics, and saves raw outputs.
│   ├── summarize_results.py   #  Aggregates raw head-level results.
│   └── plot_results.py        #  Generates manuscript figures from the summary outputs.
│
├── prompts/
│   └── prompt_set_v1.csv      #  The experimental prompt set.
│
└── results/
    ├── gpt2/                  #  csv and JSON files.
    │   ├── figures/           #  GPT-2 Figures 1-5.
    │   └── summaries/         
    │
    └── qwen_0p5b/             #  csv and JSON files.
        ├── figures/           #  Qwen Figures 1-5.
        └── summaries/         
```

## Dependencies

```bash
pip install torch transformers pandas numpy networkx matplotlib GraphRicciCurvature
```

## How to Replicate the Study

From the root directory:

### Step 1 - Generate Data

* **GPT data:**
```bash
python src/run_prompt_panel.py --prompt_csv prompts/prompt_set_v1.csv --model_name gpt2 --layers 8 11 --edge_threshold 0.01 --output_dir results/gpt2 --save_graph_payloads
```
* **Qwen data:**
```bash
python src/run_prompt_panel.py --prompt_csv prompts/prompt_set_v1.csv --model_name Qwen/Qwen2.5-0.5B --auto_n_layers 4 --edge_threshold 0.01 --output_dir results/qwen_0p5b --save_graph_payloads --device cuda --torch_dtype float16
```

**Note on Math Solver Stability:** 
Depending on your hardware and NumPy backend, the `GraphRicciCurvature` solver can sometimes hang or crash due to OpenMP/MKL thread contention (multiple CPU cores fighting over the same matrix calculations). If you experience infinite hanging during Step 1, restrict the math libraries to a single thread before running:

* **Windows (PowerShell):**
```bash
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; python src/run_prompt_panel.py --prompt_csv prompts/prompt_set_v1.csv --model_name gpt2 --layers 8 11 --edge_threshold 0.01 --output_dir results/gpt2 --save_graph_payloads
```
```bash
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; python src/run_prompt_panel.py --prompt_csv prompts/prompt_set_v1.csv --model_name Qwen/Qwen2.5-0.5B --auto_n_layers 4 --edge_threshold 0.01 --output_dir results/qwen_0p5b --save_graph_payloads --device cuda --torch_dtype float16
```
* **Mac/Linux:**
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python src/run_prompt_panel.py --prompt_csv prompts/prompt_set_v1.csv --model_name gpt2 --layers 8 11 --edge_threshold 0.01 --output_dir results/gpt2 --save_graph_payloads
```
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python src/run_prompt_panel.py --prompt_csv prompts/prompt_set_v1.csv --model_name Qwen/Qwen2.5-0.5B --auto_n_layers 4 --edge_threshold 0.01 --output_dir results/qwen_0p5b --save_graph_payloads --device cuda --torch_dtype float16
```
***
### Step 2 - Summarize Raw Data

* **Summarize GPT-2**
```bash
python src/summarize_results.py --input_csv results/gpt2/raw_head_metrics.csv --output_dir results/gpt2/summaries
```
* **Summarize Qwen**
```bash
python src/summarize_results.py --input_csv results/qwen_0p5b/raw_head_metrics.csv --output_dir results/qwen_0p5b/summaries
```
***
### Step 3 - Generate Figures

* **Plot GPT**
```bash
python src/plot_results.py --class_layer_csv results/gpt2/summaries/class_layer_summary.csv --class_layer_head_csv results/gpt2/summaries/class_layer_head_summary.csv --minimal_pair_csv results/gpt2/summaries/minimal_pair_differences.csv --graph_payload_manifest_csv results/gpt2/graph_payload_manifest.csv --output_dir results/gpt2/figures --model_label "GPT-2"
```
* **Plot Qwen**
```bash
python src/plot_results.py --class_layer_csv results/qwen_0p5b/summaries/class_layer_summary.csv --class_layer_head_csv results/qwen_0p5b/summaries/class_layer_head_summary.csv --minimal_pair_csv results/qwen_0p5b/summaries/minimal_pair_differences.csv --graph_payload_manifest_csv results/qwen_0p5b/graph_payload_manifest.csv --token_metadata_csv results/qwen_0p5b/tokenized_prompt_metadata.csv --output_dir results/qwen_0p5b/figures --model_label "Qwen 0.5B"
```
---

## Outputs

#### `run_prompt_panel.py`

Produces:

* `run_manifest.json`
* `tokenized_prompt_metadata.csv`
* `raw_head_metrics.csv`
* `prompt_level_summary.csv`
* optional graph payload JSON files
* optional `graph_payload_manifest.csv`

#### `summarize_results.py`

Produces:

* `class_layer_summary.csv`
* `class_layer_head_summary.csv`
* `prompt_layer_summary.csv`
* `minimal_pair_differences.csv`
* `summary_manifest.json`

#### `plot_results.py`

Produces:

* `fig1_class_layer_overview.png`
* `fig2_logical_minus_descriptive_heatmap.png`
* `fig3_logical_minus_structured_heatmap.png`
* `fig4_minimal_pair_effects.png`
* `fig5_topology_exemplars.png`



***

### Acknowledgments

During development, Google Gemini and OpenAI ChatGPT were used for brainstorming, structural organization, debugging assistance, and stylistic refinement. Final interpretation, analysis decisions, and manuscript claims remain the responsibility of the author.
