# Journal-revision analyses

Additional analyses produced for the journal (ACM TSC) revision of the paper,
extending the main pipeline (`../src/pipeline/step0*`). Every script here
consumes the **aggregate artifacts** written by the main pipeline (the
`artifacts/` directory) and writes small aggregate outputs; no raw data is
required or included, consistent with the rest of this repository.

## Prerequisites

Run the main pipeline first so that `artifacts/` exists (see the top-level
README), or point the scripts at an existing artifacts directory:

```bash
# option A: place this repo so that ../artifacts resolves (default), or
# option B: set the environment variable
export TSC_ART=/path/to/artifacts        # Windows: set TSC_ART=D:\...\artifacts
```

All scripts import the shared estimator from `revlib.py`, which reads
`TSC_ART` (falling back to `<repo>/artifacts`).

## Shared estimator — `revlib.py`

A self-contained re-implementation of the paper's cross-fitted matched doubly
robust (DR/AIPW) ATT estimator, validated to reproduce
`artifacts/main_effects/att_post_main.parquet` (Sample A formation reciprocity
ATT −0.185 vs. −0.187; SE 0.041 vs. 0.040; treated/control counts exact). It
exposes two estimators:

| Function | Estimator | Used as |
|---|---|---|
| `matched_aipw_att` | odds-augmented (Hajek self-normalized) AIPW; efficient-influence-function SE | primary for Sample A |
| `matched_dr_att` | outcome-adjusted matched DR (no propensity-odds term); multiplier-bootstrap SE | primary for Sample B |
| `all_estimators` | runs naive / matched-DR / odds-AIPW on one outcome | multi-estimator comparison |

The Sample B incumbent-difference outcomes are computed on minimal graphs on
which reciprocity is degenerate; there the odds-augmented term is
ill-conditioned, so `matched_dr_att` is the primary estimator and the
odds-augmented AIPW is reported as a sensitivity check.

## Scripts

| Script | What it does |
|---|---|
| `validate.py` | Reproduces the published Sample A and strict Sample B tables to validate `revlib`. |
| `a1_sampleB_expanded.py` | Mature-thread **rewiring** on the powered sample (`E_min=1`, ~2,253 treated) with the three-estimator comparison and the eligibility-ceiling probe. |
| `a_style.py` | **Content moderation**: matched formation ATT within strata of the agent reply's style (length, numeric/factual, question, …). |
| `a_topic.py` | **Topic heterogeneity** of the formation effect (exploratory keyword classifier). |
| `a_bridging.py` | Bridging: DC-BI definability, user-attribute (verified/non-verified) homophily, stance proxy. |
| `a4b_definability_fix.py` | DC-BI definability descriptives on the formed base (Table of definability). |
| `a_confounders.py` | Robustness to named confounders (topic sensitivity, controversy/visibility, poster prominence). |
| `a_edge_cutoffs.py` | Sensitivity of the formation reciprocity effect to the first-*k* human edges (*k* = 3/5/10). |
| `a7_event_study.py` | Powered edge-index event study (parallel-trends check) for the mature sample. |

## Reproducing

```bash
cd revision
python validate.py               # sanity: reproduces the published main effects
python a1_sampleB_expanded.py    # headline: powered Sample B rewiring
python a_style.py                # content moderation
python a_topic.py                # topic heterogeneity
python a_bridging.py             # bridging + definability
python a_confounders.py          # confounder robustness
python a_edge_cutoffs.py         # interaction-budget sensitivity
python a7_event_study.py         # event-study figure
```

Outputs are written under `results/` (aggregate tables, CSVs, and figures).

## Headline findings

- **Mature-thread rewiring is adequately powered** once the incumbent maturity
  bar is relaxed to `E_min=1` (~2,253 treated threads): geographic homophily
  reduction replicates as a genuine rewiring effect (−0.040, p<0.001, robust
  across estimators); the reciprocity change is directionally consistent and
  significant under the primary matched-DR estimator but estimator-sensitive.
- **The formation effect depends on reply content**: numeric/factual replies
  suppress reciprocity more strongly; longer replies dampen the branching
  increase — consistent with an information-satiation mechanism.
- The formation reciprocity effect is **stable across the first 3/5/10 human
  edges** and to controls for topic, thread visibility, and poster prominence.
