# CommentR Research Kit

> **From Solo Post to Shared Space: How Public LLM Replies Reshape Early-Stage Conversation Graphs on Weibo**
>
>
> *KDD 2026*

This repository provides an **end-to-end, reproducible** research pipeline for the paper above. It evaluates whether a single public LLM agent reply (`@CommentR`) turns one-to-many posts into many-to-many, pro-social conversations on Weibo.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Pipeline Steps](#pipeline-steps)
- [Output Artifacts](#output-artifacts)
- [Utility Tools](#utility-tools)
- [Tests](#tests)
- [Citation](#citation)
- [License](#license)

---

## Overview

Public-facing LLM agents are increasingly embedded in social platforms. This study asks whether a single public agent reply shapes the formation of subsequent **human-to-human** conversation structure within a thread. Using 216k Weibo threads involving the public agent `@CommentR`, we:

1. Match treated threads (with agent replies) to control threads on pre-anchor covariates.
2. Estimate causal effects with a **cross-fitted doubly robust (AIPW)** estimator.
3. Distinguish between **early-stage formation** (Sample A, post-only structure) and **mature-thread rewiring** (Sample B, incumbent-only pre/post deltas).

Key findings: agent replies reduce reciprocity, increase branching, and reduce geographic homophily in early-stage threads — consistent with a shift from dialogue toward broadcast-style commenting around a focal reply.

---

## Repository Structure

```
commentr_research_kit/
├── config.yaml                  # Study parameters (matching, time windows, metrics, etc.)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── src/
│   ├── pipeline/                # Sequential pipeline steps (step00–step09)
│   │   ├── step00_sanity_check.py
│   │   ├── step01_ingest_normalize.py
│   │   ├── step02_build_graphs_and_tstar.py
│   │   ├── step03_compute_outcomes.py
│   │   ├── step04_propensity_and_matching.py
│   │   ├── step05_event_study_and_did.py
│   │   ├── step06_pressure_instrument_2sri.py
│   │   ├── step07_style_features_hte.py
│   │   ├── step08_robustness_and_placebos.py
│   │   └── step09_make_tables_and_figures.py
│   │
│   ├── metrics/                 # Network & inequality metrics
│   │   ├── network_metrics.py   #   Reciprocity, branching, assortativity, DC-BI
│   │   └── gini.py              #   Gini coefficient
│   │
│   ├── models/                  # ML/NLP models
│   │   ├── prosocial.py         #   Prosocial-tone classifier (lexicon + TF-IDF/LR)
│   │   └── style.py             #   Linguistic style feature extractor
│   │
│   ├── utils/                   # Shared helpers
│   │   ├── io_utils.py          #   JSON/Parquet I/O
│   │   ├── text_utils.py        #   Chinese text feature extraction
│   │   ├── time_utils.py        #   Timezone-aware timestamp handling
│   │   └── anchor_utils.py      #   Control-thread pseudo-anchor assignment
│   │
│   └── tools/                   # Standalone CLI utilities
│       ├── make_sample.py       #   Subsample large JSON files
│       ├── compute_matched_effects.py  # Matched-pair effect differences
│       ├── show_summary.py      #   Print pipeline result summary
│       ├── train_tone_classifier.py    # Train/refresh tone classifier
│       └── sample_tone_predictions.py  # Export tone predictions for audit
│
├── tests/
│   └── test_metrics.py          # Unit tests for core metrics
│
└── artifacts/                   # Generated outputs (created by the pipeline)
    ├── ingested/                #   Normalized Parquet files
    ├── outcomes/                #   Thread-level metrics (pre/post, deltas)
    ├── matching/                #   Propensity scores, matched pairs, balance diagnostics
    ├── main_effects/            #   DR/AIPW ATT estimates (Sample A & B)
    ├── event_study/             #   Event study coefficients & DID regressions
    ├── instrument/              #   2SRI first-stage & IV results
    ├── style/                   #   Style features & HTE analysis
    ├── robustness/              #   Placebo tests & mega-thread exclusion
    ├── sample_summary/          #   Consolidated summary statistics
    ├── tone/                    #   Tone classifier model & audit samples
    └── figures/                 #   Visualization outputs (PNG)
```

---

## Requirements

- **Python** 3.10+
- **RAM** ≥ 32 GB recommended for full dataset
- **Disk** ≥ 200 GB free (intermediate Parquet files can be large)

Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `pandas`, `pyarrow`, `numpy`, `scikit-learn`, `statsmodels`, `networkx`, `matplotlib`, `scipy`, `PyYAML`, `tqdm`, `regex`, `emoji`, `joblib`.

---

## Data Preparation

This repository does **not** include the raw dataset. You need to obtain the **CommentR Interaction Dataset** separately (e.g., from the official Zenodo release) and prepare three JSON files:

| File            | Description                              |
|-----------------|------------------------------------------|
| `Users.json`    | User profiles (ID, nickname, province)   |
| `Posts.json`    | Original Weibo posts (text, timestamp)   |
| `Comments.json` | Comments/replies (text, timestamp, links) |

Place these files in a `data/` directory under the project root, or create symbolic links:

```powershell
# Example on Windows PowerShell
New-Item -ItemType SymbolicLink -Path data\users.json    -Target 'D:\CommentR Interaction Dataset\Users.json'
New-Item -ItemType SymbolicLink -Path data\posts.json    -Target 'D:\CommentR Interaction Dataset\Posts.json'
New-Item -ItemType SymbolicLink -Path data\comments.json -Target 'D:\CommentR Interaction Dataset\Comments.json'
```

```bash
# Example on Linux/macOS
ln -s /path/to/Users.json    data/users.json
ln -s /path/to/Posts.json    data/posts.json
ln -s /path/to/Comments.json data/comments.json
```

---

## Configuration

All study parameters are centralized in `config.yaml`. Key sections:

| Section | Description |
|---------|-------------|
| `agent_ids` / `agent_nickname_variants` | Identifiers for the `@CommentR` agent account |
| `time_window_minutes` | Symmetric pre/post window around anchor t* (default: 120 min) |
| `k_controls` / `caliper_logit` | Propensity matching parameters |
| `prosocial_lexicon` | Chinese lexicon for prosocial tone detection |
| `dr_*` | Cross-fitted doubly robust estimator settings |
| `min_pre_human_edges` / `min_post_human_edges` | Sample A/B split thresholds |
| `control_anchor_strategy` | Pseudo-anchor assignment for controls (`matched_median_latency`) |

Modify parameters only with documented justification. The default values reproduce the paper's results.

---

## Running the Pipeline

Run each step sequentially from the project root:

```bash
# Step 0: Validate input data
python -m src.pipeline.step00_sanity_check

# Step 1: Ingest & normalize raw JSON → Parquet (longest step)
python -m src.pipeline.step01_ingest_normalize

# Step 2: Build conversation graphs & identify agent anchors (t*)
python -m src.pipeline.step02_build_graphs_and_tstar

# Step 3*: Propensity model & matching (run BEFORE step03)
python -m src.pipeline.step04_propensity_and_matching

# Step 4*: Compute outcome metrics with matched control anchors
python -m src.pipeline.step03_compute_outcomes

# Step 5: Main DR/AIPW estimates + event study
python -m src.pipeline.step05_event_study_and_did

# Step 6: Instrumental variable (2SRI) analysis
python -m src.pipeline.step06_pressure_instrument_2sri

# Step 7: Style features & heterogeneous treatment effects
python -m src.pipeline.step07_style_features_hte

# Step 8: Robustness checks & placebo tests
python -m src.pipeline.step08_robustness_and_placebos

# Step 9: Generate summary tables & figures
python -m src.pipeline.step09_make_tables_and_figures
```

> **Note:** Steps 3 and 4 are intentionally **reordered** (run `step04` before `step03`). The matching step must execute first to provide matched control anchors for outcome computation.

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 00 | `step00_sanity_check` | Validates that required input files exist |
| 01 | `step01_ingest_normalize` | Ingests raw JSON, normalizes schemas, outputs Parquet |
| 02 | `step02_build_graphs_and_tstar` | Builds reply graphs, detects agent comments, computes t* |
| 03 | `step03_compute_outcomes` | Computes network metrics & prosocial tone for pre/post windows |
| 04 | `step04_propensity_and_matching` | Propensity score estimation, nearest-neighbor matching with caliper |
| 05 | `step05_event_study_and_did` | Cross-fitted DR/AIPW estimation (Sample A & B) + edge-index event study |
| 06 | `step06_pressure_instrument_2sri` | Two-stage residual inclusion IV using posting pressure |
| 07 | `step07_style_features_hte` | Extracts agent reply style features, trains HTE model |
| 08 | `step08_robustness_and_placebos` | Mega-thread exclusion, placebo permutation tests |
| 09 | `step09_make_tables_and_figures` | Generates publication-ready tables and figures |

### Core Metrics

- **Reciprocity (R):** Share of mutual reply dyads among human users
- **Branching Factor (BF):** Ratio of leaf to internal nodes in reply trees
- **Gini Coefficient:** Inequality of comment-count distribution
- **Assortativity:** Homophily on geographic (province) attributes
- **DC-BI (Degree-Corrected Bridging Index):** Cross-group connection rate vs. configuration model null
- **Prosocial Tone:** Lexicon + classifier-based measure of gratitude, support, and de-escalation
- **Stance Divergence / Agonism:** Low-fidelity proxy for ideological diversity

---

## Output Artifacts

All outputs are saved to `artifacts/`. Key files:

| Path | Description |
|------|-------------|
| `ingested/*.parquet` | Normalized users, posts, comments, threads |
| `outcomes/outcomes.parquet` | All per-thread metrics, pre/post deltas, tone scores |
| `matching/valid_pairs.parquet` | Strict analysis population (matched treated–control pairs) |
| `matching/balance_table.parquet` | Covariate balance diagnostics |
| `matching/weights.parquet` | ATT weights used by DR estimator |
| `main_effects/att_post_main.parquet` | **Sample A:** DR/AIPW ATT on post-only outcomes |
| `main_effects/att_main.parquet` | **Sample B:** DR/AIPW ATT on incumbent-only deltas |
| `event_study/event_coeffs_edge_main_rel.parquet` | Edge-index event study coefficients (relative to last pre-bin) |
| `event_study/did_results.parquet` | OLS robustness regressions |
| `instrument/twosri_results.parquet` | 2SRI IV estimates |
| `style/feature_importance.parquet` | Style HTE feature importance |
| `robustness/placebo_summary.parquet` | Placebo test results |
| `figures/sample/*.png` | Diagnostic and summary visualizations |

---

## Utility Tools

These standalone tools can be run independently:

```bash
# Subsample a large JSON/JSONL file (useful for development)
python -m src.tools.make_sample --input data/comments.json --ratio 0.01

# Recompute matched effects without re-fitting propensity model
python -m src.tools.compute_matched_effects

# Print a formatted summary of key results
python -m src.tools.show_summary

# Retrain the prosocial tone classifier
python -m src.tools.train_tone_classifier --comments artifacts/ingested/comments.parquet --force-retrain

# Export tone predictions for manual audit
python -m src.tools.sample_tone_predictions --comments artifacts/ingested/comments.parquet --output artifacts/tone/tone_audit.csv --n-positive 300 --n-negative 300
```

---

## Tests

Run unit tests with pytest:

```bash
pip install pytest
pytest tests/
```

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{liu2026solopost,
  title     = {From Solo Post to Shared Space: How Public LLM Replies Reshape Early-Stage Conversation Graphs on Weibo},
  author    = {Liu, Dandan and Pan, Lihu and Md Sabri, Aznul Qalid and Fan, Guangrui},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
  year      = {2026},
  address   = {Toronto, ON, Canada},
  publisher = {ACM}
}
```

---