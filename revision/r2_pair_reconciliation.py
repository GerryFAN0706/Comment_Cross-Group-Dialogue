"""R2 (round 2) — reconcile the 814,633 matched pairs with Table 2.

Reviewer 2, point 2: "the reported 814,633 matched pairs do not appear to
reconcile with the control-reuse statistics in Table 2."

They do not, as printed, because the two numbers describe two different
populations. This script documents the full accounting chain and then checks the
consequence that actually matters: whether covariate balance still holds once the
analysis is restricted to the controls that survive into estimation.

Outputs: results/r2_pairs/
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
                      "..", "results", "r2_pairs")
os.makedirs(OUTDIR, exist_ok=True)

BAL_COV = ["post_len", "post_has_question", "post_has_hashtag", "pic_num",
           "hour", "dow", "early_human_comments"]


def smd(x1, w1, x0, w0):
    m1 = np.average(x1, weights=w1)
    m0 = np.average(x0, weights=w0)
    v1 = np.average((x1 - m1) ** 2, weights=w1)
    v0 = np.average((x0 - m0) ** 2, weights=w0)
    s = np.sqrt((v1 + v0) / 2.0)
    return m1, m0, (np.nan if s == 0 else (m1 - m0) / s)


def main():
    mp = pd.read_parquet(os.path.join(ART, "matching", "matched_pairs.parquet"))
    w = pd.read_parquet(os.path.join(ART, "matching", "weights.parquet"))
    an = pd.read_parquet(os.path.join(ART, "matching", "anchors.parquet"),
                         columns=["mblogid", "anchor_source"])
    pr = pd.read_parquet(os.path.join(ART, "matching", "propensity.parquet"),
                         columns=["mblogid"])

    anchored_c = set(an.loc[an.anchor_source == "matched_median_latency", "mblogid"])
    anchored_t = set(an.loc[an.anchor_source == "tstar", "mblogid"])
    reuse = mp.groupby("mblogid_c").size()
    kt = mp.groupby("mblogid_t").size()
    surv = mp[mp["mblogid_c"].isin(anchored_c)]
    reuse_s = surv.groupby("mblogid_c").size()

    acct = dict(
        candidate_pool_propensity=int(len(pr)),
        pairs_at_matching_stage=int(len(mp)),
        distinct_treated_matched=int(mp["mblogid_t"].nunique()),
        distinct_controls_matched=int(mp["mblogid_c"].nunique()),
        controls_per_treated_mean=float(kt.mean()),
        controls_per_treated_min=int(kt.min()),
        controls_per_treated_max=int(kt.max()),
        reuse_all_mean=float(reuse.mean()), reuse_all_sd=float(reuse.std()),
        reuse_all_median=float(reuse.median()), reuse_all_p90=float(reuse.quantile(.9)),
        reuse_all_max=int(reuse.max()),
        controls_with_pseudo_anchor=int(len(anchored_c)),
        controls_dropped_no_anchor=int(mp["mblogid_c"].nunique() - len(anchored_c)),
        pairs_surviving_to_estimation=int(len(surv)),
        pairs_dropped=int(len(mp) - len(surv)),
        reuse_surv_mean=float(reuse_s.mean()), reuse_surv_sd=float(reuse_s.std()),
        treated_in_outcomes=int(len(anchored_t)),
        treated_matched_and_anchored=int(len(anchored_t & set(mp["mblogid_t"]))),
    )
    print(json.dumps(acct, indent=2), flush=True)
    with open(os.path.join(OUTDIR, "pair_accounting.json"), "w") as f:
        json.dump(acct, f, indent=2)

    # ---- balance restricted to the ESTIMATION population -------------------
    tm = R.load_thread_meta()
    wt = w[["mblogid", "w", "role"]]
    d = tm.merge(wt, on="mblogid", how="inner")
    d["T"] = (d["role"] == "treated").astype(int)
    d = d[d["mblogid"].isin(anchored_t | anchored_c)]

    # renormalised weights: within each treated set, surviving controls share
    # total weight 1 again (undoing the attrition-induced weight loss)
    kt_s = surv.groupby("mblogid_t")["mblogid_c"].size()
    pw = surv.assign(pw=1.0 / surv["mblogid_t"].map(kt_s))
    w_renorm = pw.groupby("mblogid_c")["pw"].sum().rename("w_renorm")
    d = d.merge(w_renorm, left_on="mblogid", right_index=True, how="left")
    d.loc[d["T"] == 1, "w_renorm"] = 1.0
    d["w_renorm"] = d["w_renorm"].fillna(0.0)

    rows = []
    for c in BAL_COV:
        s = d.dropna(subset=[c])
        t, k = s[s["T"] == 1], s[s["T"] == 0]
        m1a, m0a, sa = smd(t[c].astype(float).values, t["w"].values,
                           k[c].astype(float).values, k["w"].values)
        kk = k[k["w_renorm"] > 0]
        m1b, m0b, sb = smd(t[c].astype(float).values, t["w_renorm"].values,
                           kk[c].astype(float).values, kk["w_renorm"].values)
        rows.append(dict(covariate=c, n_T=len(t), n_C=len(k),
                         treated_w=m1a, control_w=m0a, smd_w=sa,
                         treated_renorm=m1b, control_renorm=m0b, smd_renorm=sb))
    bal = pd.DataFrame(rows)
    bal.to_csv(os.path.join(OUTDIR, "balance_estimation_population.csv"), index=False)
    print("\n=== post-match balance on the ESTIMATION population "
          "(treated + the 53,389 anchored controls) ===")
    print(bal.round(4).to_string(index=False), flush=True)
    print("\nmax |SMD| (as-weighted):   %.4f" % bal["smd_w"].abs().max())
    print("max |SMD| (renormalised):  %.4f" % bal["smd_renorm"].abs().max())
    print("\nWROTE ->", OUTDIR, flush=True)


if __name__ == "__main__":
    main()
