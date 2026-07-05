"""Validate the reconstructed estimator against the paper's published tables:
   att_post_main.parquet (Sample A formation) and att_main.parquet (strict Sample B).
"""
import os; os.environ["LOKY_MAX_CPU_COUNT"] = "8"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import revlib as R

ART = R.ART

def run(sample_name, group_filter, outcomes, ref_df):
    df = R.load_core(outcome_cols=outcomes)
    if sample_name == "A":
        d = df[df["sample_group"] == "early"].copy()
    else:  # strict mature Sample B (paper's Panel B): pre>=2 & post>=2 incumbent edges
        d = df[(df["n_pre_time_inc_edges"] >= 2) & (df["n_post_time_inc_edges"] >= 2)].copy()
    rows = []
    for oc in outcomes:
        r = R.matched_aipw_att(d, oc, K=5, seed=0)
        ref = ref_df[ref_df["outcome"] == oc]
        refatt = float(ref["att"].iloc[0]) if len(ref) else np.nan
        refse = float(ref["se"].iloc[0]) if len(ref) else np.nan
        rows.append(dict(outcome=oc, att=r["att"], se=r["se"], p=r["p"],
                         nT=r["n_treated"], nC=r["n_control"],
                         ref_att=refatt, ref_se=refse,
                         d_att=r["att"]-refatt))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    outsA = ["post_time_R_with_op","post_time_BF","post_time_Gini",
             "post_time_Assort_province","post_time_DCBI_province",
             "post_time_Stance_Divergence","post_time_Stance_Agonism"]
    refA = pd.read_parquet(os.path.join(ART,"main_effects","att_post_main.parquet"))
    print("=== Sample A formation (ref att_post_main) ===")
    print(run("A", None, outsA, refA).to_string(index=False,
              float_format=lambda x: f"{x:.4f}"))

    outsB = ["d_time_inc_R_with_op","d_time_inc_BF","d_time_inc_Gini",
             "d_time_inc_Assort_province","d_time_inc_DCBI_province"]
    refB = pd.read_parquet(os.path.join(ART,"main_effects","att_main.parquet"))
    print("\n=== Sample B strict E_min=2 (ref att_main) ===")
    print(run("B", None, outsB, refB).to_string(index=False,
              float_format=lambda x: f"{x:.4f}"))
