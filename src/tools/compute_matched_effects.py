import yaml
import pandas as pd
import numpy as np

from ..utils.io_utils import save_parquet, ensure_dir

ART = "artifacts/matching"


def run() -> None:
    """
    Compute matched treated-vs-control differences on outcome deltas.

    This is intentionally separated from step04 so you can run matching (step04)
    before outcomes (step03), then run this once outcomes are available, without
    retraining the propensity model.
    """
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

    # Prefer strict analysis population if available
    try:
        pairs = pd.read_parquet(f"{ART}/valid_pairs.parquet")
    except Exception:
        pairs = pd.read_parquet(f"{ART}/matched_pairs.parquet")
    outcomes = pd.read_parquet("artifacts/outcomes/outcomes.parquet")

    if pairs is None or pairs.empty or outcomes is None or outcomes.empty:
        ensure_dir(ART)
        save_parquet(pd.DataFrame(), f"{ART}/matched_effects.parquet")
        print("Matched effects skipped: missing pairs or outcomes.")
        return

    outcomes = outcomes.set_index("mblogid")
    effect_pairs = pairs[
        pairs["mblogid_t"].isin(outcomes.index) & pairs["mblogid_c"].isin(outcomes.index)
    ].copy()

    if effect_pairs.empty:
        ensure_dir(ART)
        save_parquet(pd.DataFrame(), f"{ART}/matched_effects.parquet")
        print("Matched effects skipped: no overlapping matched pairs with outcomes.")
        return

    # Prefer configured primary metrics; fall back to delta columns if missing
    core = []
    core += list(cfg.get("dr_primary_post_outcomes") or [])
    core += list(cfg.get("dr_primary_delta_outcomes") or [])
    # Backward-compatible defaults
    core += [
        "post_time_R_with_op",
        "post_time_BF",
        "post_time_Gini",
        "post_time_Assort_province",
        "post_time_DCBI_province",
        "d_time_inc_R_with_op",
        "d_time_inc_BF",
        "d_time_inc_Gini",
        "d_time_inc_Assort_province",
        "d_time_inc_DCBI_province",
        "d_prosocial_any",
        "d_prosocial_score",
        # legacy
        "d_R",
        "d_R_weighted",
        "d_BF",
        "d_BF_proxy",
        "d_Gini",
        "d_Assort_gender",
        "d_Assort_province",
        "d_DCBI_gender",
        "d_DCBI_province",
    ]
    delta_cols = [c for c in outcomes.columns if c.startswith("d_")]
    metrics = [m for m in dict.fromkeys(core) if m in outcomes.columns]
    if not metrics:
        metrics = delta_cols

    rows = []
    for metric in metrics:
        treated_vals = outcomes.loc[effect_pairs["mblogid_t"], metric]
        control_vals = outcomes.loc[effect_pairs["mblogid_c"], metric]
        df_metric = pd.DataFrame(
            {"treated": treated_vals.values, "control": control_vals.values}
        ).dropna()
        if df_metric.empty:
            continue
        diff = df_metric["treated"] - df_metric["control"]
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
        rows.append(
            {
                "metric": metric,
                "treated_mean": df_metric["treated"].mean(),
                "control_mean": df_metric["control"].mean(),
                "diff_mean": diff.mean(),
                "diff_se": se,
                "n_pairs": len(diff),
            }
        )

    matched_effects = pd.DataFrame(rows)
    ensure_dir(ART)
    save_parquet(matched_effects, f"{ART}/matched_effects.parquet")
    print(f"Matched effects saved ({len(matched_effects):,} metrics).")


if __name__ == "__main__":
    run()

