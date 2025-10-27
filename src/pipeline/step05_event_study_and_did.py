import yaml
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from ..utils.io_utils import save_parquet, ensure_dir

ART = "artifacts/event_study"


def _to_naive(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        try:
            s = s.dt.tz_convert("UTC")
        except TypeError:
            s = s.dt.tz_localize("UTC")
        s = s.dt.tz_localize(None)
    return s


def run():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    timezone = cfg.get("timezone", "UTC")
    edges = pd.read_parquet("artifacts/threads/edges.parquet")
    tstars = pd.read_parquet("artifacts/threads/tstars.parquet")
    pairs = pd.read_parquet("artifacts/matching/matched_pairs.parquet")
    outcomes = pd.read_parquet("artifacts/outcomes/outcomes.parquet")
    threads = pd.read_parquet("artifacts/ingested/threads.parquet")

    if edges.empty:
        ensure_dir(ART)
        save_parquet(pd.DataFrame(), f"{ART}/binned_edges.parquet")
        save_parquet(pd.DataFrame(), f"{ART}/did_results.parquet")
        print("Event study skipped: no edges available.")
        return
    print(f"Event study: processing {len(edges):,} edges across {edges['root_post_mblogid'].nunique():,} threads.")

    # Align edges to tau = t - t*
    tmap = dict(zip(tstars.mblogid, tstars.tstar))
    edges["tstar"] = edges["root_post_mblogid"].map(tmap)
    if not pairs.empty:
        pseudo = pairs.groupby("mblogid_c")["mblogid_t"].agg(lambda s: s.iloc[0])
        idx = edges["root_post_mblogid"].isin(pseudo.index)
        edges.loc[idx, "tstar"] = edges.loc[idx, "root_post_mblogid"].map(pseudo).map(tmap)

    edges["created_at_local"] = _to_naive(edges["created_at"])
    edges["tstar_local"] = _to_naive(edges["tstar"])
    edges["tau_min"] = (edges["created_at_local"] - edges["tstar_local"]).dt.total_seconds() / 60.0

    bins = cfg["event_bins_minutes"]
    labels = [f"[{bins[i]},{bins[i+1]})" for i in range(len(bins) - 1)]
    edges["bin"] = pd.cut(
        edges["tau_min"],
        bins=[-1e9] + bins + [1e9],
        labels=["<min"] + labels + [">=max"],
        right=False,
    )

    agg = (
        edges.groupby(["root_post_mblogid", "bin"], observed=True)
        .agg(n_edges=("u", "size"), root_replies=("is_root_reply", "sum"))
        .reset_index()
    )

    thread_flags = threads[["mblogid"]].copy()
    thread_flags["treated"] = threads["mblogid"].isin(tstars[tstars["tstar"].notna()]["mblogid"])
    agg = agg.merge(thread_flags, left_on="root_post_mblogid", right_on="mblogid", how="left")
    agg["root_reply_rate"] = np.where(agg["n_edges"] > 0, agg["root_replies"] / agg["n_edges"], np.nan)
    ensure_dir(ART)
    save_parquet(agg, f"{ART}/binned_edges.parquet")

    event_summary = (
        agg.groupby(["bin", "treated"], observed=True)
        .agg(
            mean_edges=("n_edges", "mean"),
            mean_root_reply_rate=("root_reply_rate", "mean"),
            n_threads=("root_post_mblogid", "nunique")
        )
        .reset_index()
    )
    save_parquet(event_summary, f"{ART}/event_summary.parquet")

    # Difference-in-differences on thread outcomes
    df = outcomes.copy()
    df["treated"] = df["tstar"].notna()
    for Y in [
        "R",
        "R_weighted",
        "BF",
        "BF_proxy",
        "Gini",
        "Assort_gender",
        "Assort_province",
        "DCBI_gender",
        "DCBI_province",
        "prosocial_any",
    ]:
        pre_col, post_col = f"pre_{Y}", f"post_{Y}"
        if pre_col in df.columns and post_col in df.columns:
            df[f"d_{Y}"] = df[post_col] - df[pre_col]

    threads_ct = threads[["mblogid", "created_at"]].copy()
    threads_ct["created_at_local"] = _to_naive(threads_ct["created_at"])
    df = df.merge(threads_ct[["mblogid", "created_at_local"]], on="mblogid", how="left")
    df["hour"] = df["created_at_local"].dt.hour
    df["dow"] = df["created_at_local"].dt.dayofweek

    results = []
    for col in [c for c in df.columns if c.startswith("d_")]:
        dsub = df[[col, "treated", "hour", "dow"]].dropna()
        if len(dsub) < 30 or dsub["treated"].nunique() < 2:
            continue
        model = smf.ols(f"{col} ~ treated + C(hour):C(dow)", data=dsub).fit(cov_type="HC1")
        results.append(
            {
                "outcome": col,
                "coef_treated": model.params.get("treated", float("nan")),
                "se": model.bse.get("treated", float("nan")),
                "p": model.pvalues.get("treated", float("nan")),
                "n": len(dsub),
            }
        )

    did_df = pd.DataFrame(results)
    save_parquet(did_df, f"{ART}/did_results.parquet")
    print(f"Event study complete. Saved {len(agg):,} binned rows and {len(did_df):,} DiD rows.")


if __name__ == "__main__":
    run()
