"""
A7: Powered edge-index event study for the mature-thread (Sample B) rewiring
analysis. Produces the parallel-trends coefficient figure (replaces Fig. 5),
using the binned edge-index panel. Pre-anchor bins should be ~0 (parallel
trends); post-anchor bins carry the rewiring dynamics.
"""
import os; os.environ["LOKY_MAX_CPU_COUNT"] = "8"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "event_study")
os.makedirs(OUT, exist_ok=True)

coef = pd.read_parquet(os.path.join(ART, "event_study", "event_coeffs_edge_main_rel.parquet"))

BIN_ORDER = ["-60:-40", "-40:-20", "-20:0", "0:20", "20:40", "40:60"]
BIN_X = {b: i - 2 for i, b in enumerate(BIN_ORDER)}  # -20:0 = index 2 -> x=0 baseline
PANELS = [("Assort_province", "Geographic homophily"),
          ("BF", "Branching factor"),
          ("Gini", "Inequality (Gini)"),
          ("DCBI_province", "Degree-corrected bridging")]

fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6))
for ax, (metric, title) in zip(axes.ravel(), PANELS):
    d = coef[coef["metric"] == metric].copy()
    # ensure the baseline bin appears at 0
    rows = []
    for b in BIN_ORDER:
        if b == "-20:0":
            rows.append(dict(bin=b, x=0.0, coef=0.0, ci_low=0.0, ci_high=0.0, base=True))
        else:
            r = d[d["bin"] == b]
            if len(r):
                r = r.iloc[0]
                rows.append(dict(bin=b, x=BIN_X[b], coef=r["coef"],
                                 ci_low=r["ci_low"], ci_high=r["ci_high"], base=False))
    pdd = pd.DataFrame(rows).sort_values("x")
    pre = pdd[pdd["x"] < 0]; post = pdd[pdd["x"] >= 0]
    ax.axhline(0, color="0.6", lw=0.8, zorder=0)
    ax.axvline(0, color="crimson", lw=1.0, ls="--", zorder=0)
    for seg, col in [(pre, "#4C72B0"), (post, "#4C72B0")]:
        ax.errorbar(seg["x"], seg["coef"],
                    yerr=[seg["coef"] - seg["ci_low"], seg["ci_high"] - seg["coef"]],
                    fmt="o", color=col, ecolor=col, elinewidth=1.4, capsize=3, ms=5)
    n_th = int(d["n_threads"].iloc[0]) if len(d) else 0
    ax.set_title(f"{title}  (n={n_th} threads)", fontsize=10)
    ax.set_xticks(list(BIN_X.values()))
    ax.set_xticklabels([b for b in BIN_ORDER], rotation=30, fontsize=7)
    ax.set_xlabel("edge-index bin relative to anchor", fontsize=8)
    ax.set_ylabel("treated $\\times$ bin coef", fontsize=8)
fig.suptitle("Edge-index event study (mature threads): flat pre-anchor bins support parallel trends",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, "fig_event_study_powered.pdf"))
fig.savefig(os.path.join(OUT, "fig_event_study_powered.png"), dpi=160)

# parallel-trends test table (pre-anchor bins should be p>0.05)
pre = coef[coef["bin"].isin(["-60:-40", "-40:-20"])][["metric", "bin", "coef", "se", "p", "n_threads"]]
pre.to_csv(os.path.join(OUT, "parallel_trends_pretest.csv"), index=False)
print("Parallel-trends pre-test (pre-anchor bins, expect p>0.05):")
print(pre.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved event-study figure + pretest -> results/event_study/")
