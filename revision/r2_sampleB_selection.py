"""R2 (round 2) — Sample B selection-on-a-treatment-affected-variable diagnostics.

Reviewer 2, point 4: Sample B eligibility requires post-anchor incumbent activity
>= E_min, but agent replies are shown to reduce post-anchor human-to-human edges.
Sample B membership is therefore conditioned on a treatment-affected variable.

This script produces the three things needed to answer that:
  (1) the extensive-margin ATT on Sample B eligibility itself, by arm;
  (2) an UNCONDITIONAL rewiring estimand that drops the post-anchor condition
      entirely (pre-only eligibility; boundary values for empty post graphs);
  (3) Lee (2009) trimming bounds on the conditional (published) estimand.

Outputs: results/r2_selection/
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import revlib as R

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "results", "r2_selection")
os.makedirs(OUTDIR, exist_ok=True)

SPECS = [
    # (delta col, pre col, post col, label, boundary value for empty post graph)
    ("d_time_inc_Assort_province", "pre_time_inc_Assort_province",
     "post_time_inc_Assort_province", "Assortativity (province)", 0.0),
    ("d_time_inc_R_with_op", "pre_time_inc_R_with_op",
     "post_time_inc_R_with_op", "Reciprocity R", 0.0),
    ("d_time_inc_Gini", "pre_time_inc_Gini",
     "post_time_inc_Gini", "Inequality (Gini)", 0.0),
]


def wq(x, w, q):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return float(np.interp(q, c, x))


def main():
    cols = ["n_pre_time_inc_edges", "n_post_time_inc_edges"]
    for a, b, c, _, _ in SPECS:
        cols += [a, b, c]
    df = R.load_core(outcome_cols=sorted(set(cols)))

    rows_elig, rows_unc, rows_lee = [], [], []

    for Emin in (1, 2):
        d = df[df["n_pre_time_inc_edges"] >= Emin].copy()
        d["elig"] = (d["n_post_time_inc_edges"] >= Emin).astype(float)
        dT, dC = d[d["T"] == 1], d[d["T"] == 0]
        sT = float(np.average(dT["elig"], weights=dT["w"]))
        sC = float(np.average(dC["elig"], weights=dC["w"]))
        r_dr = R.matched_dr_att(d, "elig")
        r_aw = R.matched_aipw_att(d, "elig")
        rows_elig.append(dict(E_min=Emin, n_pre_T=len(dT), n_pre_C=len(dC),
                              n_elig_T=int(dT["elig"].sum()),
                              n_elig_C=int(dC["elig"].sum()),
                              elig_rate_T=sT, elig_rate_C=sC,
                              att_dr=r_dr["att"], se_dr=r_dr["se"], p_dr=r_dr["p"],
                              att_aipw=r_aw["att"], se_aipw=r_aw["se"], p_aw=r_aw["p"]))
        print(f"[elig] E_min={Emin}: T {sT:.4f} vs C {sC:.4f} | "
              f"dr {r_dr['att']:+.4f} (p={r_dr['p']:.3g}) "
              f"aipw {r_aw['att']:+.4f} (p={r_aw['p']:.3g})", flush=True)

    # --- (2) and (3) at the main specification E_min = 1
    d = df[df["n_pre_time_inc_edges"] >= 1].copy()
    dT, dC = d[d["T"] == 1], d[d["T"] == 0]
    sT = float(np.average((dT["n_post_time_inc_edges"] >= 1).astype(float), weights=dT["w"]))
    sC = float(np.average((dC["n_post_time_inc_edges"] >= 1).astype(float), weights=dC["w"]))
    q = (sC - sT) / sC

    for dcol, precol, postcol, label, bnd in SPECS:
        dd = d.copy()
        zero = dd["n_post_time_inc_edges"] == 0
        post = pd.to_numeric(dd[postcol], errors="coerce").copy()
        post[zero] = bnd
        pre = pd.to_numeric(dd[precol], errors="coerce")
        dd["d_unc"] = post - pre
        dd = dd.dropna(subset=["d_unc"])
        u_dr = R.matched_dr_att(dd, "d_unc")
        u_aw = R.matched_aipw_att(dd, "d_unc")
        rows_unc.append(dict(outcome=label, col=dcol, boundary=bnd,
                             n_T=u_dr["n_treated"], n_C=u_dr["n_control"],
                             att_dr=u_dr["att"], se_dr=u_dr["se"], p_dr=u_dr["p"],
                             att_aipw=u_aw["att"], se_aipw=u_aw["se"], p_aw=u_aw["p"]))
        print(f"[uncond] {label}: dr {u_dr['att']:+.4f} (se {u_dr['se']:.4f}, "
              f"p={u_dr['p']:.3g}) n_T={u_dr['n_treated']} | "
              f"aipw {u_aw['att']:+.4f} (p={u_aw['p']:.3g})", flush=True)

        e = d[d["n_post_time_inc_edges"] >= 1].dropna(subset=[dcol]).copy()
        ctrl = e[e["T"] == 0]
        x = pd.to_numeric(ctrl[dcol], errors="coerce").values
        wc = ctrl["w"].values
        lo_cut, hi_cut = wq(x, wc, q), wq(x, wc, 1 - q)
        v = pd.to_numeric(e[dcol], errors="coerce").values
        keep_hi = e[(e["T"] == 1) | ((e["T"] == 0) & (v <= hi_cut))]
        keep_lo = e[(e["T"] == 1) | ((e["T"] == 0) & (v >= lo_cut))]
        base = R.matched_dr_att(e, dcol)
        b_hi = R.matched_dr_att(keep_hi, dcol)
        b_lo = R.matched_dr_att(keep_lo, dcol)
        rows_lee.append(dict(outcome=label, col=dcol, trim_q=q,
                             sel_rate_T=sT, sel_rate_C=sC,
                             att_point=base["att"], se_point=base["se"],
                             lower=b_lo["att"], upper=b_hi["att"],
                             n_T=base["n_treated"], n_C=base["n_control"]))
        print(f"[lee]    {label}: q={q:.3f} point {base['att']:+.4f} "
              f"-> [{b_lo['att']:+.4f}, {b_hi['att']:+.4f}]", flush=True)

    pd.DataFrame(rows_elig).to_csv(os.path.join(OUTDIR, "sampleB_eligibility_att.csv"), index=False)
    pd.DataFrame(rows_unc).to_csv(os.path.join(OUTDIR, "sampleB_unconditional.csv"), index=False)
    pd.DataFrame(rows_lee).to_csv(os.path.join(OUTDIR, "sampleB_lee_bounds.csv"), index=False)
    print("\nWROTE ->", OUTDIR, flush=True)


if __name__ == "__main__":
    main()
