import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from ..utils.io_utils import ensure_dir


def load_data(threads_path: Path, outcomes_path: Path, pairs_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threads = pd.read_parquet(threads_path)
    outcomes = pd.read_parquet(outcomes_path)
    pairs = pd.read_parquet(pairs_path)
    return threads, outcomes, pairs


def bootstrap_diff(treated: pd.Series, control: pd.Series, n_bootstrap: int = 1000, seed: int = 2025) -> tuple[float, tuple[float, float]]:
    diff = treated.mean() - control.mean()
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        t_sample = treated.sample(frac=1, replace=True, random_state=rng.integers(0, 1e9))
        c_sample = control.sample(frac=1, replace=True, random_state=rng.integers(0, 1e9))
        boot.append(t_sample.mean() - c_sample.mean())
    boot = np.sort(boot)
    lower = np.percentile(boot, 2.5)
    upper = np.percentile(boot, 97.5)
    return diff, (lower, upper)


def main():
    parser = argparse.ArgumentParser(description="Compute equity audit metrics by subgroup.")
    parser.add_argument("--threads", type=Path, default=Path("artifacts/ingested/threads.parquet"))
    parser.add_argument("--outcomes", type=Path, default=Path("artifacts/outcomes/outcomes.parquet"))
    parser.add_argument("--pairs", type=Path, default=Path("artifacts/matching/matched_pairs.parquet"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/equity/equity_audit.parquet"))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    threads, outcomes, pairs = load_data(args.threads, args.outcomes, args.pairs)
    ensure_dir(args.output.parent.as_posix())

    # Reply probability: treated vs control after matching
    treated_threads = set(pairs["mblogid_t"])
    control_threads = set(pairs["mblogid_c"])

    outcome_subset = outcomes.set_index("mblogid")
    metrics = []

    strata = [
        ("verification", threads, "op_id", "user", lambda u: u.get("verified", False)),
        ("follower_decile_low_high", threads, "op_id", "user", lambda u: u.get("followers_count_decile", 0) <= 3 if u else False),
        ("province_macro", threads, "op_id", "user", lambda u: u.get("macro_region", "unknown")),
    ]

    for name, df, id_col, user_col, func in strata:
        treated_ids = [tid for tid in treated_threads if tid in df["mblogid"].values]
        control_ids = [cid for cid in control_threads if cid in df["mblogid"].values]
        if not treated_ids or not control_ids:
            continue
        treated_out = outcome_subset.loc[treated_ids]
        control_out = outcome_subset.loc[control_ids]
        diff = treated_out["all_prosocial_any"].mean() - control_out["all_prosocial_any"].mean()
        diff_boot, (lower, upper) = bootstrap_diff(treated_out["all_prosocial_any"], control_out["all_prosocial_any"], args.n_bootstrap, args.seed)
        metrics.append({
            "stratum": name,
            "metric": "prosocial_any",
            "diff": diff_boot,
            "ci_lower": lower,
            "ci_upper": upper
        })
    out_df = pd.DataFrame(metrics)
    out_df.to_parquet(args.output, index=False)
    print(f"Saved equity audit summary to {args.output}")


if __name__ == "__main__":
    main()
