# CommentR Conversation Bridging Study — Research Kit

This code provides an **end-to-end, reproducible** pipeline to evaluate whether a single AI agent reply (@CommentR) turns one-to-many posts into many-to-many, pro-social conversations on Weibo.

## Code includes:
- **Step-by-step pipeline** scripts in `src/pipeline/` — run them sequentially.
- **Core metrics** (Reciprocity, Branching, Gini, Assortativity, DC-BI, Prosocial tone).
- **Causal identification** modules: Propensity matching, Event study, 2SRI instrument.
- **Style HTE** estimators and mediation scaffolding.
- **Robustness & falsifications**.
- **Artifacts** are saved per step under `artifacts/` with checksums for reproducibility.

## Quickstart
1. Install Python 3.10+ and dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your JSON inputs in `data/`:
   - `userspro.json` (list or JSONL)
   - `postspro.json`
   - `commentspro.json`
3. Edit `config.yaml` — set the agent account numeric ID(s) if you have them.
4. Run the pipeline:
   ```bash
   python -m src.pipeline.step00_sanity_check
   python -m src.pipeline.step01_ingest_normalize
   python -m src.pipeline.step02_build_graphs_and_tstar
   python -m src.pipeline.step03_compute_outcomes
   python -m src.pipeline.step04_propensity_and_matching
   python -m src.pipeline.step05_event_study_and_did
   python -m src.pipeline.step06_pressure_instrument_2sri
   python -m src.pipeline.step07_style_features_hte
   python -m src.pipeline.step08_robustness_and_placebos
   python -m src.pipeline.step09_make_tables_and_figures
   ```

## Outputs (artifacts/)
- `ingested/` normalized parquet files
- `threads/` per-thread metadata and edges
- `outcomes/` metrics per thread (pre/post and event-study bins)
- `matching/` propensity scores, matched sets, diagnostics
- `event_study/` coefficients, pre-trends tests
- `instrument/` first-stage & 2SRI results
- `style/` extracted features, HTE surfaces
- `robustness/` placebo and leave-one-province-out results
- `figures/` publication-ready PNG/SVG

See inline docstrings and comments for each step.
