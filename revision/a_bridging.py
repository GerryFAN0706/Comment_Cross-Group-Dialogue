"""
a_bridging.py — Bridging analysis + Table 15 (DC-BI definability descriptives).

Addresses R2-4, R1-3, AE-3. Table 15 is a MUST-FIX (submitted PDF had
literal PLACEHOLDER values).

Part 1: Table 15 — DC-BI definability descriptives in Sample A matched pop.
Part 2: user-attribute bridging (formation Assort_province ATT by OP verified).
Part 3: stance proxy (formation + rewiring stance effects).
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import revlib as R

OUTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "bridging"))
os.makedirs(OUTDIR, exist_ok=True)

import revlib, os
MP_PATH = os.path.join(revlib.ART, "matching", "matched_pairs.parquet")
USERS_PATH = os.path.join(revlib.ART, "ingested", "users.parquet")

DCBI = "post_time_DCBI_province"
ASSORT = "post_time_Assort_province"
RECIP = "post_time_R_with_op"
STANCE_F = "post_time_Stance_Divergence"
STANCE_R = "d_time_inc_Stance_Divergence"

# covariates for definable-vs-nondefinable comparison
COVS = ["post_len", "post_has_question", "post_has_hashtag", "pic_num",
        "n_comments", "n_post_time_edges", "early_human_comments"]


def std_diff(a, b):
    """Standardized mean difference between two groups (pooled SD)."""
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt((va + vb) / 2.0)
    if sp == 0 or np.isnan(sp):
        return 0.0 if ma == mb else np.nan
    return (ma - mb) / sp


def main():
    print("Loading core...")
    df = R.load_core(outcome_cols=[DCBI, ASSORT, RECIP, STANCE_F, STANCE_R])
    # Sample A = 'early'
    A = df[df["sample_group"] == "early"].copy()
    print(f"Sample A (early) rows: {len(A)}  treated={int((A['T']==1).sum())} control={int((A['T']==0).sum())}")

    # ---------------- SANITY ----------------
    sanity = {}
    print("\n=== SANITY CHECKS ===")
    rec = R.all_estimators(A, RECIP)
    print("Reciprocity (formation):")
    print(rec.to_string())
    sanity["reciprocity"] = rec.set_index("estimator")["att"].to_dict()

    asr = R.all_estimators(A, ASSORT)
    print("\nAssortativity(province) (formation):")
    print(asr.to_string())
    sanity["assortativity"] = asr.set_index("estimator")["att"].to_dict()

    dr_rec = sanity["reciprocity"].get("matched_dr")
    aipw_rec = sanity["reciprocity"].get("odds_aipw")
    dr_as = sanity["assortativity"].get("matched_dr")
    ok_rec = (dr_rec is not None) and (abs(dr_rec - (-0.167)) < 0.03)
    ok_aipw = (aipw_rec is not None) and (abs(aipw_rec - (-0.185)) < 0.04)
    ok_as = (dr_as is not None) and (abs(dr_as - (-0.05)) < 0.03)
    sanity_passed = bool(ok_rec and ok_as)
    print(f"\nSanity: recip_dr={dr_rec} (ref -0.167, ok={ok_rec}) | "
          f"recip_aipw={aipw_rec} (ref -0.185, ok={ok_aipw}) | "
          f"assort_dr={dr_as} (ref -0.05, ok={ok_as})")
    print(f"SANITY PASSED = {sanity_passed}")

    # ================= PART 1: TABLE 15 =================
    print("\n=== PART 1: Table 15 definability ===")
    # Table 15 describes the FORMATION analysis population: threads in the matched
    # pairs that have a DEFINED post-window formation structure (post_time_R_with_op
    # non-null <=> n_post_time_edges>0). DC-BI (post_time_DCBI_province) is then
    # DEFINABLE within that base only when the thread has sufficient cross-group
    # degree mass (non-null DCBI). Definability rate = P(DCBI defined | formed).
    mp_all = pd.read_parquet(MP_PATH)
    early_ids = set(A["mblogid"])
    mpA = mp_all[mp_all["mblogid_t"].isin(early_ids)]
    t_pair_ids = set(mpA["mblogid_t"])
    c_pair_ids = set(mpA["mblogid_c"])
    # matched-pair population, restricted to the formed (formation-defined) base
    base = A[(((A["T"] == 1) & (A["mblogid"].isin(t_pair_ids))) |
              ((A["T"] == 0) & (A["mblogid"].isin(c_pair_ids)))) &
             (A[RECIP].notna())].copy()
    base["definable"] = base[DCBI].notna()
    print(f"Formation base (formed & matched): treated={int((base['T']==1).sum())} "
          f"control={int((base['T']==0).sum())}")

    # definability rate treated vs control (within formed base)
    rate_t = base.loc[base["T"] == 1, "definable"].mean()
    rate_c = base.loc[base["T"] == 0, "definable"].mean()
    n_t = int((base["T"] == 1).sum())
    n_c = int((base["T"] == 0).sum())
    n_def_t = int(base.loc[base["T"] == 1, "definable"].sum())
    n_def_c = int(base.loc[base["T"] == 0, "definable"].sum())
    print(f"Definability rate: treated={rate_t:.4f} ({n_def_t}/{n_t}), "
          f"control={rate_c:.4f} ({n_def_c}/{n_c})")
    def_rates = pd.DataFrame([
        {"group": "treated", "n_total": n_t, "n_definable": n_def_t, "definability_rate": rate_t},
        {"group": "control", "n_total": n_c, "n_definable": n_def_c, "definability_rate": rate_c},
    ])
    def_rates.to_csv(os.path.join(OUTDIR, "definability_rates.csv"), index=False)

    # covariate comparison: definable vs non-definable (within formed base)
    rows = []
    grp_def = base[base["definable"]]
    grp_non = base[~base["definable"]]
    for cov in COVS:
        if cov not in base.columns:
            rows.append({"covariate": cov, "mean_definable": np.nan,
                         "mean_nondefinable": np.nan, "std_diff": np.nan,
                         "n_definable": np.nan, "n_nondefinable": np.nan,
                         "note": "column missing"})
            continue
        vd = pd.to_numeric(grp_def[cov], errors="coerce")
        vn = pd.to_numeric(grp_non[cov], errors="coerce")
        rows.append({
            "covariate": cov,
            "mean_definable": vd.mean(),
            "mean_nondefinable": vn.mean(),
            "std_diff": std_diff(vd, vn),
            "n_definable": int(vd.notna().sum()),
            "n_nondefinable": int(vn.notna().sum()),
            "note": "",
        })
    table15 = pd.DataFrame(rows)
    print("\nTable 15 (definable vs non-definable covariates):")
    print(table15.to_string())
    table15.to_csv(os.path.join(OUTDIR, "table15_definability.csv"), index=False)
    def_rates.to_parquet(os.path.join(OUTDIR, "definability_rates.parquet"), index=False)
    table15.to_parquet(os.path.join(OUTDIR, "table15_definability.parquet"), index=False)

    # ================= PART 2: user-attribute bridging =================
    print("\n=== PART 2: Assort_province ATT by OP verified ===")
    users = pd.read_parquet(USERS_PATH, columns=["_id", "verified", "follower_decile", "mbrank"])
    users = users.rename(columns={"_id": "op_id"})
    # op_id types
    A["op_id"] = A["op_id"].astype(str)
    users["op_id"] = users["op_id"].astype(str)
    Au = A.merge(users, on="op_id", how="left")
    ver_cov = Au["verified"].notna().mean()
    print(f"OP verified merge coverage: {ver_cov:.3f}")

    mp = pd.read_parquet(MP_PATH)

    part2_rows = []
    for vflag, label in [(True, "verified_True"), (False, "verified_False")]:
        treated_ids = set(Au.loc[(Au["T"] == 1) & (Au["verified"] == vflag), "mblogid"])
        if len(treated_ids) == 0:
            part2_rows.append({"split": label, "estimator": "matched_dr", "att": np.nan,
                               "se": np.nan, "p": np.nan, "n_treated": 0, "n_control": 0})
            continue
        ctrl_ids = set(mp.loc[mp.mblogid_t.isin(treated_ids), "mblogid_c"])
        sub = df[((df["T"] == 1) & (df["mblogid"].isin(treated_ids))) |
                 ((df["T"] == 0) & (df["mblogid"].isin(ctrl_ids)))]
        est = R.all_estimators(sub, ASSORT)
        est["split"] = label
        est["n_treated_stratum"] = len(treated_ids)
        print(f"\n{label}: n_treated={len(treated_ids)}")
        print(est.to_string())
        for _, r in est.iterrows():
            part2_rows.append({"split": label, "estimator": r["estimator"],
                               "att": r["att"], "se": r["se"], "p": r["p"],
                               "n_treated": int(r["n_treated"]), "n_control": int(r["n_control"])})
    part2 = pd.DataFrame(part2_rows)
    part2.to_csv(os.path.join(OUTDIR, "assort_by_verified.csv"), index=False)
    part2.to_parquet(os.path.join(OUTDIR, "assort_by_verified.parquet"), index=False)

    # ================= PART 3: stance proxy =================
    print("\n=== PART 3: stance proxy ===")
    stance_rows = []
    for oc, lab in [(STANCE_F, "formation_post_time"), (STANCE_R, "rewiring_d_time_inc")]:
        est = R.all_estimators(A, oc)
        est["measure"] = lab
        est["outcome"] = oc
        print(f"\nStance {lab} ({oc}):")
        print(est.to_string())
        stance_rows.append(est)
    stance = pd.concat(stance_rows, ignore_index=True)
    stance.to_csv(os.path.join(OUTDIR, "stance_effects.csv"), index=False)
    stance.to_parquet(os.path.join(OUTDIR, "stance_effects.parquet"), index=False)

    # ================= summary.json =================
    summary = {
        "sanity_passed": sanity_passed,
        "sanity": {
            "reciprocity_matched_dr": dr_rec,
            "reciprocity_odds_aipw": aipw_rec,
            "assortativity_matched_dr": dr_as,
            "refs": {"recip_dr": -0.167, "recip_aipw": -0.185, "assort_dr": -0.05},
        },
        "part1_definability": {
            "rate_treated": float(rate_t), "rate_control": float(rate_c),
            "n_treated": n_t, "n_control": n_c,
            "n_definable_treated": n_def_t, "n_definable_control": n_def_c,
            "paper_claim": {"treated": 0.313, "control": 0.297},
            "table15": table15.to_dict(orient="records"),
        },
        "part2_assort_by_verified": part2.to_dict(orient="records"),
        "part2_verified_merge_coverage": float(ver_cov),
        "part3_stance": stance[["measure", "outcome", "estimator", "att", "se", "p",
                                "n_treated", "n_control"]].to_dict(orient="records"),
    }
    with open(os.path.join(OUTDIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=lambda o: None if pd.isna(o) else float(o))

    # SUMMARY.md
    def dr_row(estdf, est="matched_dr"):
        r = estdf[estdf.estimator == est].iloc[0]
        return f"att={r['att']:.4f}, se={r['se']:.4f}, p={r['p']:.4g}, n_t={int(r['n_treated'])}"

    with open(os.path.join(OUTDIR, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("# Bridging + Table 15\n\n")
        f.write(f"Sanity passed: **{sanity_passed}**\n\n")
        f.write("## Sanity\n")
        f.write(f"- Reciprocity matched_dr={dr_rec:.4f} (ref -0.167); odds_aipw={aipw_rec} (ref -0.185)\n")
        f.write(f"- Assortativity(province) matched_dr={dr_as:.4f} (ref -0.05)\n\n")
        f.write("## Part 1: Table 15 DC-BI definability (Sample A matched pop)\n")
        f.write(f"- Definability rate treated = **{rate_t*100:.1f}%** ({n_def_t}/{n_t}) [paper ~31.3%]\n")
        f.write(f"- Definability rate control = **{rate_c*100:.1f}%** ({n_def_c}/{n_c}) [paper ~29.7%]\n\n")
        f.write(table15.to_markdown(index=False))
        f.write("\n\n## Part 2: Formation Assort_province ATT by OP verified\n")
        f.write(f"- OP verified merge coverage: {ver_cov:.3f}\n")
        f.write(f"- Province is a COARSE macro-group; this is a partial finer-grained bridging check.\n\n")
        f.write(part2[part2.estimator == "matched_dr"].to_markdown(index=False))
        f.write("\n\n## Part 3: Stance proxy (low-fidelity ideological proxy)\n")
        for lab, oc in [("formation_post_time", STANCE_F), ("rewiring_d_time_inc", STANCE_R)]:
            sub = stance[stance.measure == lab]
            f.write(f"- {lab} ({oc}): {dr_row(sub)}\n")

    print(f"\nAll outputs saved to {OUTDIR}")
    return summary


if __name__ == "__main__":
    main()
