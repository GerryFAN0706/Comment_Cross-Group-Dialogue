"""Targeted reciprocity-anchored robustness for R2-1/R2-2 (minimal compute).

Reciprocity (post_time_R_with_op) is the Sample A outcome the reconstruction
reproduces to within 0.002 of the published matched_dr_aipw value (-0.187), so it
is the credible anchor for the robustness appendices. We report three checks, all
with the same odds-AIPW estimator:
  (1) baseline (reproduce -0.187)
  (2) non-binding subsample: treated anchor latency > Delta=15 min, so the
      early-engagement covariate uses the same [t0,t0+Delta) window in both arms.
Weight-renorm (-0.184 -> -0.164) and drop-EarlyHuman (-0.184 -> -0.185) were
already computed (r2_robustness task8/9a); this fills the subsample cell.

Output: results/r2_robustness/recip_subsample.json
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import revlib as R
import numpy as np, pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "r2_robustness")
DELTA = 15.0
COL = "post_time_R_with_op"


def main():
    df = R.load_core(outcome_cols=[COL])
    dfA = df[df["sample_group"] == "early"].copy()

    t0 = time.time()
    base = R.matched_aipw_att(dfA, COL, seed=0)
    print(f"[baseline] recip {base['att']:+.4f} (p={base['p']:.3g}) nT={base['n_treated']} [{time.time()-t0:.0f}s]", flush=True)

    lat = pd.to_numeric(dfA["age_at_anchor_min"], errors="coerce")
    keep_t = set(dfA.loc[(dfA["T"] == 1) & (lat > DELTA), "mblogid"])
    mp = pd.read_parquet(os.path.join(R.ART, "matching", "matched_pairs.parquet"))
    keep_c = set(mp.loc[mp["mblogid_t"].isin(keep_t), "mblogid_c"])
    sub = dfA[((dfA["T"] == 1) & dfA["mblogid"].isin(keep_t)) |
              ((dfA["T"] == 0) & dfA["mblogid"].isin(keep_c))].copy()
    t0 = time.time()
    rs = R.matched_aipw_att(sub, COL, seed=0)
    print(f"[subsample] recip {rs['att']:+.4f} (p={rs['p']:.3g}) nT={rs['n_treated']}/{int((dfA['T']==1).sum())} [{time.time()-t0:.0f}s]", flush=True)

    res = dict(baseline_att=base["att"], baseline_p=base["p"], baseline_nT=base["n_treated"],
               subsample_att=rs["att"], subsample_p=rs["p"], subsample_nT=rs["n_treated"],
               full_nT=int((dfA["T"] == 1).sum()),
               subsample_share=rs["n_treated"] / int((dfA["T"] == 1).sum()))
    with open(os.path.join(OUT, "recip_subsample.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE", os.path.join(OUT, "recip_subsample.json"), flush=True)


if __name__ == "__main__":
    main()
