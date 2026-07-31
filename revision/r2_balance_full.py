"""R2 (round 2) — reconcile Table 3 (main text) with Appendix Table 13.

Reviewer 2, point 3: "the covariate values in Table 3 differ substantially from
Appendix Table 13 despite both being described as diagnostics for the strict
matched population."

Table 3 in the manuscript reproduces artifacts/matching/balance_table.parquet
exactly. Appendix Table 13 lists a DIFFERENT covariate set (OP follower decile,
verified, account age, statuses count, topic PC1/PC2) with values that contradict
Table 3 on the two shared rows. Those OP-level variables are not in the matching
feature set at all.

This script produces the honest replacement:
  (A) the complete, verbatim balance table actually used by the matching
      (all 8 matched covariates) -> the corrected Appendix Table 13; and
  (B) a genuine *post-match* balance diagnostic for the OP-level covariates that
      were NOT matched on (verified, follower decile, mbrank, log followers,
      account age if available), computed on the strict matched population with
      matching weights, so the manuscript can state what is actually true about
      them instead of the current unsupported claim.

Outputs: results/r2_balance/
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import revlib as R

ART = R.ART
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "results", "r2_balance")
os.makedirs(OUTDIR, exist_ok=True)

OP_COLS = ["op_verified", "op_mbrank", "op_follower_decile", "op_log_followers"]


def smd(x1, w1, x0, w0):
    """Standardized mean difference with matching weights (pooled SD)."""
    m1 = np.average(x1, weights=w1)
    m0 = np.average(x0, weights=w0)
    v1 = np.average((x1 - m1) ** 2, weights=w1)
    v0 = np.average((x0 - m0) ** 2, weights=w0)
    s = np.sqrt((v1 + v0) / 2.0)
    return m1, m0, (np.nan if s == 0 else (m1 - m0) / s)


def main():
    # ---------- (A) the real, complete matching balance table ----------
    bt = pd.read_parquet(os.path.join(ART, "matching", "balance_table.parquet"))
    bt.to_csv(os.path.join(OUTDIR, "matching_balance_verbatim.csv"), index=False)
    print("=== (A) artifacts/matching/balance_table.parquet (this IS main-text Table 3) ===")
    print(bt.round(4).to_string(index=False), flush=True)
    print(f"\n  n covariates in the matching feature set: {len(bt)}", flush=True)

    # ---------- (B) OP-level covariates: matched but never balanced on ----------
    df = R.load_core(outcome_cols=["n_pre_time_inc_edges"])
    u = pd.read_parquet(os.path.join(ART, "ingested", "users.parquet"),
                        columns=["_id", "verified", "mbrank", "follower_decile",
                                 "followers_count", "created_at"])
    u = u.rename(columns={"_id": "op_id"})
    u["op_verified"] = pd.to_numeric(u["verified"], errors="coerce").astype(float)
    u["op_mbrank"] = pd.to_numeric(u["mbrank"], errors="coerce")
    u["op_follower_decile"] = pd.to_numeric(u["follower_decile"], errors="coerce")
    u["op_log_followers"] = np.log1p(
        pd.to_numeric(u["followers_count"], errors="coerce").clip(lower=0))
    keep = ["op_id"] + OP_COLS
    df = df.merge(u[keep].drop_duplicates("op_id"), on="op_id", how="left")

    rows = []
    for c in OP_COLS:
        d = df.dropna(subset=[c])
        t, k = d[d["T"] == 1], d[d["T"] == 0]
        # unweighted = "as matched, before weighting"; weighted = ATT matching weights
        m1u, m0u, su = smd(t[c].values, np.ones(len(t)), k[c].values, np.ones(len(k)))
        m1w, m0w, sw = smd(t[c].values, t["w"].values, k[c].values, k["w"].values)
        rows.append(dict(covariate=c, n_T=len(t), n_C=len(k),
                         treated_unw=m1u, control_unw=m0u, smd_unw=su,
                         treated_wtd=m1w, control_wtd=m0w, smd_wtd=sw))
    ob = pd.DataFrame(rows)
    ob.to_csv(os.path.join(OUTDIR, "op_covariate_balance_matched.csv"), index=False)
    print("\n=== (B) OP-level covariates on the strict matched population "
          "(NOT in the matching feature set) ===")
    print(ob.round(4).to_string(index=False), flush=True)
    print("\nWROTE ->", OUTDIR, flush=True)


if __name__ == "__main__":
    main()
