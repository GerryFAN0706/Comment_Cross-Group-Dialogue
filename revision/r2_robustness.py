"""R2 (round 2) — three de-risking robustness analyses for the revision.

Task 8  Weight renormalization: after the zero-comment controls drop out, do the
        Sample A headline ATTs move if we renormalize surviving control weights
        within each treated set (restores max |SMD| 0.415 -> 0.110)?

Task 9  Censoring-asymmetry robustness for the early-engagement covariate
        (treated censored at t*, controls use the full Delta window):
        (a) drop early_human_comments from the outcome model;
        (b) restrict to the non-binding subsample (treated anchor latency > Delta),
            where the covariate is the full-Delta window in BOTH arms.

Task 10 Second boundary convention for the unconditional Sample B estimand:
        empty post-incumbent graph -> Delta = 0 (no rewiring) instead of A_post = 0.

Outputs: results/r2_robustness/
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import revlib as R

ART = R.ART
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "results", "r2_robustness")
os.makedirs(OUTDIR, exist_ok=True)

DELTA = 15.0  # minutes, matches Eq. (5)
SAMPLE_A = {
    "Reciprocity R": "post_time_R_with_op",
    "Branching BF": "post_time_BF",
    "Assortativity A (prov)": "post_time_Assort_province",
    "DC-BI (prov)": "post_time_DCBI_province",
}
PANEL_B = [
    ("d_time_inc_Assort_province", "pre_time_inc_Assort_province",
     "post_time_inc_Assort_province", "Assortativity (province)"),
    ("d_time_inc_R_with_op", "pre_time_inc_R_with_op",
     "post_time_inc_R_with_op", "Reciprocity R"),
    ("d_time_inc_Gini", "pre_time_inc_Gini",
     "post_time_inc_Gini", "Inequality (Gini)"),
]


def renorm_weights_early(dfA):
    """Renormalize control weights so surviving controls share total weight 1
    within each treated set, over the early (Sample A) population."""
    mp = pd.read_parquet(os.path.join(ART, "matching", "matched_pairs.parquet"))
    et = set(dfA.loc[dfA["T"] == 1, "mblogid"])
    ec = set(dfA.loc[dfA["T"] == 0, "mblogid"])
    m = mp[mp["mblogid_t"].isin(et) & mp["mblogid_c"].isin(ec)].copy()
    kt = m.groupby("mblogid_t")["mblogid_c"].transform("size")
    m["pw"] = 1.0 / kt
    wr = m.groupby("mblogid_c")["pw"].sum().rename("w_renorm")
    out = dfA.merge(wr, left_on="mblogid", right_index=True, how="left")
    out.loc[out["T"] == 1, "w_renorm"] = 1.0
    out["w_renorm"] = out["w_renorm"].fillna(0.0)
    return out


def task8(df_all):
    dfA = df_all[df_all["sample_group"] == "early"].copy()
    dfA = renorm_weights_early(dfA)
    rows = []
    for label, col in SAMPLE_A.items():
        base = R.matched_aipw_att(dfA, col, seed=0)
        d2 = dfA.copy()
        d2["w"] = d2["w_renorm"]
        d2 = d2[d2["w"] > 0]
        reno = R.matched_aipw_att(d2, col, seed=0)
        rows.append(dict(outcome=label, col=col,
                         att_published=base["att"], se_published=base["se"], p_published=base["p"],
                         att_renorm=reno["att"], se_renorm=reno["se"], p_renorm=reno["p"],
                         delta=reno["att"] - base["att"],
                         n_T=base["n_treated"], n_C=base["n_control"]))
        print(f"[task8] {label:24s} published {base['att']:+.4f} (p={base['p']:.3g}) "
              f"-> renorm {reno['att']:+.4f} (p={reno['p']:.3g})  delta {reno['att']-base['att']:+.4f}",
              flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "task8_weight_renorm.csv"), index=False)
    return rows


def task9(df_all):
    dfA = df_all[df_all["sample_group"] == "early"].copy()
    # (a) drop early_human_comments from the OUTCOME model covariates
    orig = list(R.COV_NUM)
    no_eh = [c for c in orig if c != "early_human_comments"]
    rows_a = []
    for label, col in SAMPLE_A.items():
        base = R.matched_aipw_att(dfA, col, seed=0)
        R.COV_NUM = no_eh
        try:
            drop = R.matched_aipw_att(dfA, col, seed=0)
        finally:
            R.COV_NUM = orig
        rows_a.append(dict(outcome=label, col=col,
                           att_baseline=base["att"], p_baseline=base["p"],
                           att_no_earlyhuman=drop["att"], p_no_earlyhuman=drop["p"],
                           delta=drop["att"] - base["att"],
                           n_T=base["n_treated"]))
        print(f"[task9a] {label:24s} baseline {base['att']:+.4f} "
              f"-> drop-EarlyHuman {drop['att']:+.4f}  delta {drop['att']-base['att']:+.4f}",
              flush=True)
    pd.DataFrame(rows_a).to_csv(os.path.join(OUTDIR, "task9a_drop_earlyhuman.csv"), index=False)

    # (b) non-binding subsample: treated anchor latency > Delta (covariate is the
    #     full-Delta window in both arms). Keep those treated + controls matched to them.
    lat = pd.to_numeric(dfA["age_at_anchor_min"], errors="coerce")
    keep_t = set(dfA.loc[(dfA["T"] == 1) & (lat > DELTA), "mblogid"])
    mp = pd.read_parquet(os.path.join(ART, "matching", "matched_pairs.parquet"))
    keep_c = set(mp.loc[mp["mblogid_t"].isin(keep_t), "mblogid_c"])
    sub = dfA[((dfA["T"] == 1) & dfA["mblogid"].isin(keep_t)) |
              ((dfA["T"] == 0) & dfA["mblogid"].isin(keep_c))].copy()
    n_t_full = int((dfA["T"] == 1).sum())
    n_t_sub = int((sub["T"] == 1).sum())
    rows_b = []
    for label, col in SAMPLE_A.items():
        base = R.matched_aipw_att(dfA, col, seed=0)
        r = R.matched_aipw_att(sub, col, seed=0)
        rows_b.append(dict(outcome=label, col=col,
                           att_full=base["att"], p_full=base["p"],
                           att_subsample=r["att"], se_subsample=r["se"], p_subsample=r["p"],
                           n_T_full=n_t_full, n_T_sub=r["n_treated"]))
        print(f"[task9b] {label:24s} full {base['att']:+.4f} "
              f"-> latency>{DELTA:.0f}min {r['att']:+.4f} (p={r['p']:.3g}, nT={r['n_treated']})",
              flush=True)
    print(f"[task9b] non-binding subsample retains {n_t_sub}/{n_t_full} treated "
          f"({100*n_t_sub/n_t_full:.1f}%)", flush=True)
    pd.DataFrame(rows_b).to_csv(os.path.join(OUTDIR, "task9b_nonbinding_subsample.csv"), index=False)
    return dict(a=rows_a, b=rows_b, sub_treated=n_t_sub, full_treated=n_t_full)


def task10(df_all):
    """Unconditional Sample B with Delta=0 boundary (no-rewiring) for empty post."""
    d = df_all[df_all["n_pre_time_inc_edges"] >= 1].copy()
    rows = []
    for dcol, precol, postcol, label in PANEL_B:
        dd = d.copy()
        zero = dd["n_post_time_inc_edges"] == 0
        # convention 2: empty post-incumbent graph => no rewiring => delta 0
        delta = pd.to_numeric(dd[dcol], errors="coerce").copy()
        delta[zero] = 0.0
        dd["d_bnd0"] = delta
        dd = dd.dropna(subset=["d_bnd0"])
        r_dr = R.matched_dr_att(dd, "d_bnd0")
        r_aw = R.matched_aipw_att(dd, "d_bnd0")
        rows.append(dict(outcome=label, col=dcol, convention="delta=0 (no rewiring)",
                         att_dr=r_dr["att"], se_dr=r_dr["se"], p_dr=r_dr["p"],
                         att_aipw=r_aw["att"], se_aipw=r_aw["se"], p_aw=r_aw["p"],
                         n_T=r_dr["n_treated"], n_C=r_dr["n_control"]))
        print(f"[task10] {label:24s} dr {r_dr['att']:+.4f} (p={r_dr['p']:.3g}) | "
              f"aipw {r_aw['att']:+.4f} (p={r_aw['p']:.3g})", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "task10_boundary_convention2.csv"), index=False)
    return rows


def main():
    need = list(SAMPLE_A.values())
    for a, b, c, _ in PANEL_B:
        need += [a, b, c]
    df_all = R.load_core(outcome_cols=sorted(set(need)))
    print("loaded core:", df_all.shape, flush=True)
    r8 = task8(df_all)
    print()
    r9 = task9(df_all)
    print()
    r10 = task10(df_all)
    with open(os.path.join(OUTDIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(task8=r8, task9=r9, task10=r10), f, indent=2)
    print("\nWROTE ->", OUTDIR, flush=True)


if __name__ == "__main__":
    main()
