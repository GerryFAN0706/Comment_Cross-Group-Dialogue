"""
a_style.py — Content/agent-reply STYLE MODERATION of the Sample A formation effect.
Addresses reviewers R1-4, AE-4, R3-add-2.

Question: does the Sample A formation effect on reciprocity (post_time_R_with_op)
and branching (post_time_BF) depend on HOW the agent replied?

Design: restrict to sample_group=='early'. For each style feature, split TREATED
into two strata (binary by value 0/1; continuous len_chars by median). For each
stratum build subsample = (treated in stratum) + (their matched controls) and
estimate matched_dr ATT. Report per-stratum ATT and the moderation difference.
Secondary: within-treated OLS interaction cross-check.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import revlib as R

ART = r"D:\research  tasks\TSC revision\artifacts"
OUTDIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "results", "style"))
os.makedirs(OUTDIR, exist_ok=True)

OUTCOMES = ["post_time_R_with_op", "post_time_BF"]
FEATURES = ["len_chars", "has_question", "has_enumeration", "has_numbers",
            "has_time_markers", "empathy", "politeness", "hedges"]
# len_chars is continuous -> median split; rest are binary 0/1
CONT = {"len_chars"}
SEED = 0

# ------------------------------------------------------------------ load
df = R.load_core(outcome_cols=OUTCOMES)
sf = pd.read_parquet(os.path.join(ART, "style", "style_features.parquet"))
mp = pd.read_parquet(os.path.join(ART, "matching", "matched_pairs.parquet"))

early = df[df["sample_group"] == "early"].copy()
early_treated_ids = set(early.loc[early["T"] == 1, "mblogid"])

# Attach style to the early treated set (style exists only for treated)
sf_e = sf[sf["mblogid"].isin(early_treated_ids)].copy()
print(f"early treated={len(early_treated_ids)}  style-matched treated={len(sf_e)}")

# ------------------------------------------------------------------ sanity
sanity = R.matched_dr_att(early, "post_time_R_with_op", seed=SEED)
print(f"SANITY full-early recip matched_dr att={sanity['att']:.4f} "
      f"(ref ~ -0.167) se={sanity['se']:.4f} nT={sanity['n_treated']}")
SANITY_OK = abs(sanity["att"] - (-0.167)) < 0.01

# ------------------------------------------------------------------ helper
def stratum_att(treated_ids, outcome):
    """Build subsample = treated in stratum + their matched controls; matched_dr."""
    treated_ids = set(treated_ids)
    ctrl_ids = set(mp.loc[mp.mblogid_t.isin(treated_ids), "mblogid_c"])
    sub = df[((df["T"] == 1) & (df["mblogid"].isin(treated_ids))) |
             ((df["T"] == 0) & (df["mblogid"].isin(ctrl_ids)))].copy()
    r = R.matched_dr_att(sub, outcome, seed=SEED)
    return r

# ------------------------------------------------------------------ moderation loop
rows = []
for feat in FEATURES:
    if feat in CONT:
        med = sf_e[feat].median()
        lo_ids = sf_e.loc[sf_e[feat] <= med, "mblogid"]
        hi_ids = sf_e.loc[sf_e[feat] > med, "mblogid"]
        strata = [(f"{feat}<=median({med:g})", lo_ids),
                  (f"{feat}>median({med:g})", hi_ids)]
    else:
        s0 = sf_e.loc[sf_e[feat] == 0, "mblogid"]
        s1 = sf_e.loc[sf_e[feat] == 1, "mblogid"]
        strata = [(f"{feat}=0", s0), (f"{feat}=1", s1)]

    for outcome in OUTCOMES:
        stratum_res = {}
        for label, ids in strata:
            r = stratum_att(ids, outcome)
            stratum_res[label] = r
            rows.append(dict(feature=feat, outcome=outcome, stratum=label,
                             n_treated_style=int(len(ids)),
                             att=r["att"], se=r["se"], p=r["p"],
                             n_treated=r["n_treated"], n_control=r["n_control"]))
        # moderation difference = ATT(stratum1) - ATT(stratum0)
        (l0, r0), (l1, r1) = list(stratum_res.items())
        diff = r1["att"] - r0["att"]
        se_diff = np.sqrt((r0["se"] or np.nan) ** 2 + (r1["se"] or np.nan) ** 2)
        z = diff / se_diff if se_diff and se_diff > 0 else np.nan
        from scipy import stats as _st
        pdiff = float(2 * _st.norm.sf(abs(z))) if np.isfinite(z) else np.nan
        rows.append(dict(feature=feat, outcome=outcome,
                         stratum=f"DIFF[{l1} - {l0}]",
                         n_treated_style=None,
                         att=diff, se=se_diff, p=pdiff,
                         n_treated=None, n_control=None))
        print(f"{feat:16s} {outcome:20s} {l0}: {r0['att']:+.4f}  "
              f"{l1}: {r1['att']:+.4f}  diff={diff:+.4f} (p={pdiff:.3f})")

mod_df = pd.DataFrame(rows)
mod_df.to_parquet(os.path.join(OUTDIR, "moderation_att.parquet"), index=False)
mod_df.to_csv(os.path.join(OUTDIR, "moderation_att.csv"), index=False)

# ------------------------------------------------------------------ OLS interaction cross-check
# Within-treated OLS: regress outcome on style features (treated with defined outcome).
import statsmodels.api as sm
ols_out = {}
tr = early[early["T"] == 1][["mblogid", "post_time_R_with_op", "post_time_BF"]]
tr = tr.merge(sf_e, on="mblogid", how="inner")
for outcome in OUTCOMES:
    d = tr.dropna(subset=[outcome]).copy()
    X = d[FEATURES].astype(float).copy()
    # standardize len_chars for interpretable coef alongside binaries
    X["len_chars"] = (X["len_chars"] - X["len_chars"].mean()) / X["len_chars"].std()
    X = sm.add_constant(X)
    y = d[outcome].astype(float)
    m = sm.OLS(y, X).fit(cov_type="HC1")
    tbl = pd.DataFrame({"coef": m.params, "se": m.bse, "t": m.tvalues,
                        "p": m.pvalues}).reset_index().rename(columns={"index": "term"})
    tbl["outcome"] = outcome
    tbl["n"] = int(len(d))
    ols_out[outcome] = tbl
    print(f"\nOLS {outcome} (n={len(d)}):")
    print(tbl.to_string(index=False))

ols_df = pd.concat(ols_out.values(), ignore_index=True)
ols_df.to_parquet(os.path.join(OUTDIR, "ols_interaction.parquet"), index=False)
ols_df.to_csv(os.path.join(OUTDIR, "ols_interaction.csv"), index=False)

# ------------------------------------------------------------------ summaries
summary = dict(
    analysis="style_moderation",
    sample="sample_group=='early' (Sample A formation)",
    n_early_treated=int(len(early_treated_ids)),
    n_style_treated=int(len(sf_e)),
    sanity=dict(full_early_recip_matched_dr_att=float(sanity["att"]),
                se=float(sanity["se"]), n_treated=int(sanity["n_treated"]),
                reference=-0.167, passed=bool(SANITY_OK)),
    features=FEATURES,
    outcomes=OUTCOMES,
    n_moderation_rows=int(len(mod_df)),
)
with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# SUMMARY.md
lines = ["# Style moderation of Sample A formation effect\n",
         f"Sample: {summary['sample']}",
         f"Early treated: {summary['n_early_treated']}; with style features: {summary['n_style_treated']}\n",
         f"**Sanity**: full-early reciprocity matched_dr att = {sanity['att']:.4f} "
         f"(ref -0.167) -> {'PASS' if SANITY_OK else 'FAIL'}\n",
         "## Per-stratum matched_dr ATT (moderation)\n"]
disp = mod_df[~mod_df.stratum.str.startswith("DIFF")].copy()
for feat in FEATURES:
    for outcome in OUTCOMES:
        sub = mod_df[(mod_df.feature == feat) & (mod_df.outcome == outcome)]
        lines.append(f"### {feat} on {outcome}")
        for _, rr in sub.iterrows():
            lines.append(f"- {rr['stratum']}: att={rr['att']:+.4f} "
                         f"se={rr['se'] if pd.notna(rr['se']) else float('nan'):.4f} "
                         f"p={rr['p'] if pd.notna(rr['p']) else float('nan'):.3f} "
                         f"(nT={rr['n_treated'] if pd.notna(rr['n_treated']) else '-'})")
        lines.append("")
lines.append("## OLS within-treated interaction cross-check\n")
for outcome in OUTCOMES:
    lines.append(f"### {outcome}")
    lines.append(ols_out[outcome].to_string(index=False))
    lines.append("")
with open(os.path.join(OUTDIR, "SUMMARY.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\nSaved to", OUTDIR)
