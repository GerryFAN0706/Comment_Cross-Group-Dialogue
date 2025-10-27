import os, yaml, pandas as pd, numpy as np
from ..utils.io_utils import save_parquet, ensure_dir

ART = "artifacts/robustness"

def run():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    outcomes = pd.read_parquet("artifacts/outcomes/outcomes.parquet")
    threads = pd.read_parquet("artifacts/ingested/threads.parquet")

    # Excluding mega-threads (top 1% by edges)
    q = outcomes["n_edges"].quantile(cfg["max_thread_percentile_exclude"]/100.0)
    excl = outcomes[outcomes["n_edges"]>q]["mblogid"]
    sub = outcomes[~outcomes["mblogid"].isin(excl)]
    print(f"Robustness: excluded top {cfg['max_thread_percentile_exclude']}% threads ({len(excl):,}) by edges.")
    # Simple summaries comparing with/without exclusion
    cols = [c for c in outcomes.columns if c.startswith("d_")]
    sum_full = outcomes[cols].mean().rename("full")
    sum_sub = sub[cols].mean().rename("exclude_top1")
    df = pd.concat([sum_full, sum_sub], axis=1)
    ensure_dir(ART)
    save_parquet(df.reset_index().rename(columns={"index":"metric"}), f"{ART}/exclude_mega_threads.parquet")

    # Placebo pseudo-replies: random t* for untreated threads
    untreated = outcomes[outcomes["tstar"].isna()]["mblogid"].tolist()
    # For a simple placebo, we shuffle pre/post labels within untreated (will be refined in full pipeline)
    placebo = outcomes[outcomes["mblogid"].isin(untreated)].copy()
    print(f"Placebo sample size: {len(placebo):,} untreated threads.")
    for c in [c for c in outcomes.columns if c.startswith(("pre_","post_"))]:
        placebo[c] = np.random.permutation(placebo[c].values)
    # Compute deltas
    for Y in ["R","BF","Gini","Assort_gender","DCBI_gender","prosocial_any"]:
        pre_col, post_col = f"pre_{Y}", f"post_{Y}"
        if pre_col in placebo.columns and post_col in placebo.columns:
            placebo[f"d_{Y}"] = placebo[post_col] - placebo[pre_col]
    placebo_summ = placebo[[c for c in placebo.columns if c.startswith("d_")]].mean().rename("placebo").reset_index().rename(columns={"index":"metric"})

    save_parquet(placebo_summ, f"{ART}/placebo_summary.parquet")
    print("Robustness and placebo summaries saved.")

if __name__ == "__main__":
    run()
