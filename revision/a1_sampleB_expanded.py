"""
A1 (HEADLINE): Strengthened Sample B — rewiring ATT on the expanded mature
sample (E_min=1) vs the strict cell (E_min=2), full cross-fitted matched
DR/AIPW. Answers AE-1, R2-1/2, R3-add-1. Also probes the eligibility ceiling.

Design: incumbent-only difference-in-differences. Outcomes are the pre->post
change among the FIXED incumbent participant set (d_time_inc_*), which is a
genuine rewiring estimand (participant composition held constant).
"""
import os; os.environ["LOKY_MAX_CPU_COUNT"] = "8"
import warnings; warnings.filterwarnings("ignore")
import json, numpy as np, pandas as pd
import revlib as R

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "results", "sampleB")
os.makedirs(OUTDIR, exist_ok=True)

OUTCOMES = ["d_time_inc_R_with_op", "d_time_inc_BF", "d_time_inc_Gini",
            "d_time_inc_Assort_province", "d_time_inc_DCBI_province",
            "d_time_inc_Stance_Divergence"]
LABEL = {"d_time_inc_R_with_op": "Reciprocity R",
         "d_time_inc_BF": "Branching BF",
         "d_time_inc_Gini": "Inequality (Gini)",
         "d_time_inc_Assort_province": "Assortativity (province)",
         "d_time_inc_DCBI_province": "DC-BI (province)",
         "d_time_inc_Stance_Divergence": "Stance divergence"}

def eligibility_ceiling(df):
    """How many treated threads clear each maturity rule (why strict was null)."""
    t = df[df["T"] == 1]
    rows = []
    for emin in (1, 2, 3):
        n = int(((t["n_pre_time_inc_edges"] >= emin) & (t["n_post_time_inc_edges"] >= emin)).sum())
        rows.append(dict(rule=f"incumbent, pre>={emin} & post>={emin}", n_treated=n))
    return pd.DataFrame(rows)

def run_sample(df, emin):
    d = df[(df["n_pre_time_inc_edges"] >= emin) & (df["n_post_time_inc_edges"] >= emin)].copy()
    frames = []
    for oc in OUTCOMES:
        t = R.all_estimators(d, oc, seed=0)
        t["label"] = LABEL[oc]; t["E_min"] = emin
        frames.append(t)
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    df = R.load_core(outcome_cols=OUTCOMES)
    df = df[df["propensity"].notna()].copy()

    ceil = eligibility_ceiling(df)
    print("=== Eligibility ceiling (treated) ===")
    print(ceil.to_string(index=False))

    res = {}
    for emin in (2, 1):
        r = run_sample(df, emin)
        res[emin] = r
        print(f"\n=== Sample B rewiring ATT, E_min={emin}: three estimators ===")
        show = r.pivot_table(index="label", columns="estimator",
                             values=["att", "p"], aggfunc="first")
        print(show.to_string(float_format=lambda x: f"{x:.4f}"))

    allres = pd.concat(res.values(), ignore_index=True)
    allres.to_parquet(os.path.join(OUTDIR, "sampleB_rewiring_att.parquet"))
    ceil.to_parquet(os.path.join(OUTDIR, "eligibility_ceiling.parquet"))

    # headline table (E_min=1) with the stable matched-DR primary + odds-AIPW sensitivity
    e1 = allres[allres["E_min"] == 1]
    hdr = []
    for o in OUTCOMES:
        dr = e1[(e1.outcome == o) & (e1.estimator == "matched_dr")].iloc[0]
        aw = e1[(e1.outcome == o) & (e1.estimator == "odds_aipw")].iloc[0]
        nv = e1[(e1.outcome == o) & (e1.estimator == "naive_matched")].iloc[0]
        hdr.append(dict(outcome=LABEL[o], n_treated=int(dr["n_treated"]),
                        naive_att=nv["att"],
                        dr_att=dr["att"], dr_se=dr["se"], dr_p=dr["p"],
                        aipw_att=aw["att"], aipw_se=aw["se"], aipw_p=aw["p"]))
    hdrdf = pd.DataFrame(hdr)
    hdrdf.to_csv(os.path.join(OUTDIR, "sampleB_headline_Emin1.csv"), index=False)
    print("\n=== HEADLINE (E_min=1): stable matched-DR primary vs odds-AIPW sensitivity ===")
    print(hdrdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump({"ceiling": ceil.to_dict("records"),
                   "results": allres.to_dict("records")}, f, indent=2, default=float)
    print("\nSaved -> results/sampleB/")
