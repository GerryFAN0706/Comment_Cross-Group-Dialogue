import yaml
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pandas.api.types import DatetimeTZDtype
from ..utils.io_utils import save_parquet, ensure_dir

ART = "artifacts/instrument"


def _to_naive(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if isinstance(s.dtype, DatetimeTZDtype):
        try:
            s = s.dt.tz_convert("UTC")
        except TypeError:
            s = s.dt.tz_localize("UTC")
        s = s.dt.tz_localize(None)
    return s


def run():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    posts = pd.read_parquet("artifacts/ingested/posts.parquet")
    tstars = pd.read_parquet("artifacts/threads/tstars.parquet")
    outcomes = pd.read_parquet("artifacts/outcomes/outcomes.parquet")

    posts["created_at_local"] = _to_naive(posts["created_at"])
    posts["minute"] = posts["created_at_local"].dt.floor("min")
    Q = posts.groupby("minute").size().rename("count").reset_index()
    w = cfg["pressure_window_minutes"]
    if Q.empty:
        Q["Qm"] = pd.Series(dtype=float)
    else:
        Q = Q.set_index("minute").sort_index()
        Q["Qm"] = (
            Q["count"]
            .rolling(f"{w*2 + 1}min", center=True)
            .sum()
            .bfill()
            .ffill()
        )
        Q = Q.reset_index()
    print(f"Pressure index computed for {len(Q):,} minute bins.")

    threads = posts[["mblogid", "created_at_local"]].drop_duplicates().rename(
        columns={"created_at_local": "post_time"}
    )
    threads["minute"] = threads["post_time"].dt.floor("min")
    threads = threads.merge(Q[["minute", "Qm"]], on="minute", how="left")
    treated_ids = set(tstars[tstars["tstar"].notna()]["mblogid"])
    threads["treated"] = threads["mblogid"].isin(treated_ids).astype(int)
    threads["hour"] = threads["post_time"].dt.hour
    threads["dow"] = threads["post_time"].dt.dayofweek

    d1 = threads.dropna(subset=["Qm"]).copy()
    print(f"Instrument stage: {len(d1):,} observations after dropping missing Qm.")
    first_stage_summary = pd.DataFrame(
        {"param": ["coef_Qm", "p_Qm", "n"], "value": [np.nan, np.nan, len(d1)]}
    )
    resid_col = None

    if not d1.empty and d1["treated"].nunique() > 1 and d1["Qm"].nunique() > 1:
        try:
            mod1 = smf.logit("treated ~ Qm + C(hour):C(dow)", data=d1).fit(disp=False)
            d1["resid1"] = mod1.resid_response
            resid_col = "resid1"
            first_stage_summary = pd.DataFrame(
                {
                    "param": ["coef_Qm", "p_Qm", "n"],
                    "value": [
                        mod1.params.get("Qm", float("nan")),
                        mod1.pvalues.get("Qm", float("nan")),
                        len(d1),
                    ],
                }
            )
            print("First-stage logit converged.")
        except Exception as exc:  # pragma: no cover
            print(f"First-stage logit failed: {exc}")
    else:
        print("Instrument stage skipped: insufficient variation in treatment or pressure.")

    res2 = []
    if resid_col:
        df = outcomes.merge(
            d1[["mblogid", "treated", resid_col, "hour", "dow"]], on="mblogid", how="inner"
        )
        for Y in ["d_R", "d_BF", "d_Gini", "d_DCBI_gender", "d_DCBI_province", "d_prosocial_any"]:
            if Y not in df.columns:
                continue
            dsub = df[[Y, "treated", resid_col, "hour", "dow"]].dropna()
            if len(dsub) < 30 or dsub["treated"].nunique() < 2:
                continue
            mod2 = smf.ols(
                f"{Y} ~ treated + {resid_col} + C(hour):C(dow)", data=dsub
            ).fit(cov_type="HC1")
            res2.append(
                {
                    "outcome": Y,
                    "coef_treated": mod2.params.get("treated", float("nan")),
                    "se": mod2.bse.get("treated", float("nan")),
                    "p": mod2.pvalues.get("treated", float("nan")),
                    "n": len(dsub),
                }
            )
    res2 = pd.DataFrame(res2)

    ensure_dir(ART)
    save_parquet(Q, f"{ART}/pressure_series.parquet")
    save_parquet(first_stage_summary, f"{ART}/first_stage_summary.parquet")
    save_parquet(res2, f"{ART}/twosri_results.parquet")
    print(f"2SRI estimation complete. Saved {len(res2):,} second-stage rows.")


if __name__ == "__main__":
    run()
