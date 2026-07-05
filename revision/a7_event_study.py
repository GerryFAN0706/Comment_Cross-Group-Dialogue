"""
A7: Powered edge-index event study for the mature-thread (Sample B) rewiring
analysis. Produces the parallel-trends coefficient figure, styled to match the
paper's preferred publication aesthetic (sans-serif, thin axes, a/b/c/d panel
labels, no in-figure title -- all detail lives in the LaTeX caption).
"""
import os; os.environ["LOKY_MAX_CPU_COUNT"] = "8"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

import revlib
ART = revlib.ART
OUT = os.path.join(os.path.dirname(__file__), "results", "event_study")
os.makedirs(OUT, exist_ok=True)


def configure_style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


BLUE = "#3B6EA5"
RED = "#C0392B"
GREY = "#8A8A8A"
BIN_ORDER = ["-60:-40", "-40:-20", "-20:0", "0:20", "20:40", "40:60"]
BIN_X = {b: i - 2 for i, b in enumerate(BIN_ORDER)}   # baseline bin -20:0 at x=0
PANELS = [("Assort_province", "Geographic homophily"),
          ("BF", "Branching factor"),
          ("Gini", "Inequality (Gini)"),
          ("DCBI_province", "Degree-corrected bridging")]


def mm(x):
    return x / 25.4


def main():
    configure_style()
    coef = pd.read_parquet(os.path.join(ART, "event_study", "event_coeffs_edge_main_rel.parquet"))

    fig, axes = plt.subplots(2, 2, figsize=(mm(178), mm(112)))
    for k, (ax, (metric, title)) in enumerate(zip(axes.ravel(), PANELS)):
        d = coef[coef["metric"] == metric]
        rows = []
        for b in BIN_ORDER:
            if b == "-20:0":
                rows.append(dict(x=0.0, coef=0.0, lo=0.0, hi=0.0, base=True))
            else:
                r = d[d["bin"] == b]
                if len(r):
                    r = r.iloc[0]
                    rows.append(dict(x=BIN_X[b], coef=r["coef"], lo=r["ci_low"],
                                     hi=r["ci_high"], base=False))
        p = pd.DataFrame(rows).sort_values("x")
        ax.axhline(0, color=GREY, lw=0.8, zorder=0)
        ax.axvline(0, color=RED, lw=1.0, ls=(0, (4, 3)), zorder=0)
        base = p[p["base"]]; pts = p[~p["base"]]
        ax.errorbar(pts["x"], pts["coef"],
                    yerr=[pts["coef"] - pts["lo"], pts["hi"] - pts["coef"]],
                    fmt="o", ms=4.5, color=BLUE, ecolor=BLUE, elinewidth=1.1,
                    capsize=2.5, capthick=1.1, zorder=3)
        ax.plot(base["x"], base["coef"], "o", ms=4.5, mfc="white",
                mec=BLUE, mew=1.2, zorder=3)  # open marker at the omitted bin
        ax.set_xticks(list(BIN_X.values()))
        ax.set_xticklabels(BIN_ORDER, rotation=30, ha="right")
        ax.set_title(title, pad=4, fontsize=9)
        # bold panel letter, top-left, outside the axes
        ax.text(-0.20, 1.06, "abcd"[k], transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left")
        if k % 2 == 0:
            ax.set_ylabel(r"treated $\times$ bin coefficient")
        if k >= 2:
            ax.set_xlabel("edge-index bin (relative to anchor)")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.13, hspace=0.55, wspace=0.28)
    fig.savefig(os.path.join(OUT, "fig_event_study_powered.pdf"))
    fig.savefig(os.path.join(OUT, "fig_event_study_powered.png"), dpi=170)

    pre = coef[coef["bin"].isin(["-60:-40", "-40:-20"])][["metric", "bin", "coef", "se", "p", "n_threads"]]
    pre.to_csv(os.path.join(OUT, "parallel_trends_pretest.csv"), index=False)
    print("Parallel-trends pre-test (pre-anchor bins, expect p>0.05):")
    print(pre.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved styled event-study figure -> results/event_study/")


if __name__ == "__main__":
    main()
