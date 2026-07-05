"""
A4b: Corrected DC-BI definability (Table 15). The paper's 31.3%/29.7% rates
condition on the FORMED base (threads with post-window human structure), not
on all treated threads. Recompute on that base + covariate contrast.
"""
import os; os.environ["LOKY_MAX_CPU_COUNT"] = "8"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import revlib as R

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "bridging")

df = R.load_core(outcome_cols=["post_time_DCBI_province", "post_time_R_with_op",
                               "post_time_Assort_province"])
d = df[df["sample_group"] == "early"].copy()
# "formed base": threads with at least one post-window human edge (structure to speak of)
d["formed"] = d["n_post_time_edges"] >= 1
d["dcbi_def"] = d["post_time_DCBI_province"].notna()

rows = []
for grp, sub in [("treated", d[d["T"] == 1]), ("control", d[d["T"] == 0])]:
    base = sub[sub["formed"]]
    rate_formed = base["dcbi_def"].mean()
    rate_all = sub["dcbi_def"].mean()
    rows.append(dict(group=grp, n_all=len(sub), n_formed=int(sub["formed"].sum()),
                     n_definable=int(sub["dcbi_def"].sum()),
                     definability_rate_formed_base=round(rate_formed, 4),
                     definability_rate_all_threads=round(rate_all, 4)))
rate = pd.DataFrame(rows)
rate.to_csv(os.path.join(OUT, "definability_rates_CORRECTED.csv"), index=False)
print("=== Corrected DC-BI definability (paper reports 31.3% treated / 29.7% control) ===")
print(rate.to_string(index=False))

# Table 15: covariate contrast, definable vs non-definable, WITHIN the formed base
base = d[d["formed"]].copy()
covs = ["post_len", "post_has_question", "post_has_hashtag", "pic_num",
        "n_comments", "n_post_time_edges", "early_human_comments"]
t15 = []
for c in covs:
    a = base.loc[base["dcbi_def"], c].astype(float)
    b = base.loc[~base["dcbi_def"], c].astype(float)
    sd = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12)
    t15.append(dict(covariate=c, mean_definable=round(a.mean(), 3),
                    mean_nondefinable=round(b.mean(), 3), std_diff=round(sd, 3),
                    n_definable=len(a), n_nondefinable=len(b)))
t15 = pd.DataFrame(t15)
t15["note"] = "descriptive; n_post_time_edges/n_comments/early_human_comments are post/peri-treatment"
t15.to_csv(os.path.join(OUT, "table15_definability_CORRECTED.csv"), index=False)
print("\n=== Table 15 (formed base; definable vs non-definable DC-BI) ===")
print(t15[["covariate", "mean_definable", "mean_nondefinable", "std_diff"]].to_string(index=False))
print("\nSaved corrected definability -> results/bridging/*_CORRECTED.csv")
