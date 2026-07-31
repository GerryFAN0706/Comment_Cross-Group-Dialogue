"""R2 (round 2) — resolve the two P0 verification questions in R2_REVISION_PLAN.md.

Q1. Why do 59,971 of the 113,360 matched controls receive no pseudo-anchor?
    Hypothesis in the plan: they carry no comments, hence no reply graph, hence
    no anchor/window/outcome. Verify against thread_meta / ingested threads.

Q2. What is `early_human_comments`, exactly, per arm?
    Hypothesis: treated = censored at min(t0+Delta, t*); controls = fixed
    [t0, t0+Delta) window (no censoring at the pseudo-anchor). Verify by
    binning early_human_comments against anchor latency per arm.

Outputs: results/r2_pairs/drop_and_censoring.txt
"""
import os
import sys
import io
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import revlib as R

ART = R.ART
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "results", "r2_pairs")
os.makedirs(OUTDIR, exist_ok=True)

buf = io.StringIO()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    buf.write(s + "\n")


def main():
    mp = pd.read_parquet(os.path.join(ART, "matching", "matched_pairs.parquet"))
    an = pd.read_parquet(os.path.join(ART, "matching", "anchors.parquet"))
    say("anchors.parquet columns:", list(an.columns))
    tm = pd.read_parquet(os.path.join(ART, "threads", "thread_meta.parquet"))
    say("thread_meta columns:", list(tm.columns))

    anchored_c = set(an.loc[an.anchor_source == "matched_median_latency", "mblogid"])
    matched_c = set(mp["mblogid_c"])
    dropped = matched_c - anchored_c
    say(f"\nmatched controls: {len(matched_c):,}; anchored: {len(anchored_c):,}; "
        f"dropped: {len(dropped):,}")

    # ---- Q1: where do the dropped controls disappear? ----------------------
    in_tm = dropped & set(tm["mblogid"])
    say(f"dropped controls present in thread_meta: {len(in_tm):,} "
        f"({100*len(in_tm)/len(dropped):.1f}%)")

    if len(in_tm):
        sub = tm[tm["mblogid"].isin(in_tm)]
        say("dropped-and-in-thread_meta: n_comments describe:")
        say(sub["n_comments"].describe().round(3).to_string())
        say("share with n_comments == 0: %.4f" % (sub["n_comments"] == 0).mean())

    # thread-level graph records
    try:
        th = pd.read_parquet(os.path.join(ART, "ingested", "threads.parquet"))
        say("\ningested/threads columns:", list(th.columns))
        say(f"dropped controls present in ingested/threads: "
            f"{len(dropped & set(th['mblogid'])):,}" if "mblogid" in th.columns else "no mblogid col")
    except Exception as e:
        say("ingested/threads read failed:", e)

    # comment counts from the raw comments table for a sample of dropped ids
    try:
        cm = pd.read_parquet(os.path.join(ART, "ingested", "comments.parquet"),
                             columns=["root_post_mblogid"])
        vc = cm["root_post_mblogid"].value_counts()
        drop_l = list(dropped)
        have = vc.reindex(drop_l).fillna(0)
        say("\nraw comment count for DROPPED matched controls:")
        say("  share with 0 comments: %.4f" % (have == 0).mean())
        say("  mean %.3f | p50 %.1f | p90 %.1f | max %d"
            % (have.mean(), have.median(), have.quantile(.9), have.max()))
        surv_l = list(anchored_c)
        have_s = vc.reindex(surv_l).fillna(0)
        say("raw comment count for SURVIVING (anchored) controls:")
        say("  share with 0 comments: %.4f" % (have_s == 0).mean())
        say("  mean %.3f | p50 %.1f | p90 %.1f | max %d"
            % (have_s.mean(), have_s.median(), have_s.quantile(.9), have_s.max()))
    except Exception as e:
        say("comments read failed:", e)

    # ---- Q2: early_human_comments censoring, per arm -----------------------
    say("\n" + "=" * 72)
    say("Q2: early_human_comments vs anchor latency (Delta = 15 min)")
    a2 = an.copy()
    t0 = None
    for c in ("created_at", "thread_created_at", "t0"):
        if c in tm.columns:
            t0 = c
            break
    m = tm[["mblogid", t0, "early_human_comments", "n_comments"]].merge(
        an[["mblogid", "anchor_source", "anchor_time"]], on="mblogid", how="inner")
    m["lat_min"] = (pd.to_datetime(m["anchor_time"]) -
                    pd.to_datetime(m[t0])).dt.total_seconds() / 60.0
    bins = [-0.001, 0.5, 1, 2, 5, 10, 15, 30, 60, np.inf]
    m["lat_bin"] = pd.cut(m["lat_min"], bins)
    for src, lab in (("tstar", "TREATED (anchor = t*)"),
                     ("matched_median_latency", "CONTROL (anchor = pseudo t~*)")):
        g = (m[m.anchor_source == src]
             .groupby("lat_bin", observed=True)
             .agg(n=("mblogid", "size"),
                  early_mean=("early_human_comments", "mean"),
                  early_max=("early_human_comments", "max")))
        say(f"\n{lab}: early_human_comments by anchor latency")
        say(g.round(3).to_string())

    # If censored at anchor, threads with lat < 0.5 min can contain almost no
    # comments before the anchor. Controls violating that => not censored.
    v = m[(m.anchor_source == "matched_median_latency") & (m.lat_min < 0.5)]
    say(f"\ncontrols with pseudo-anchor < 0.5 min: n={len(v):,}, "
        f"early_human mean={v['early_human_comments'].mean():.3f}, "
        f"max={v['early_human_comments'].max():.0f}"
        f"  -> censoring at the pseudo-anchor is "
        f"{'VIOLATED (fixed-window)' if v['early_human_comments'].max() > 3 else 'plausible'}")
    vt = m[(m.anchor_source == "tstar") & (m.lat_min < 0.5)]
    say(f"treated with t* < 0.5 min:              n={len(vt):,}, "
        f"early_human mean={vt['early_human_comments'].mean():.3f}, "
        f"max={vt['early_human_comments'].max():.0f}")

    # ---- weight mass + the 37 unmatched treated ----------------------------
    say("\n" + "=" * 72)
    w = pd.read_parquet(os.path.join(ART, "matching", "weights.parquet"))
    wc = w[w["role"] != "treated"]
    surv_mask = wc["mblogid"].isin(anchored_c)
    say(f"control weight mass total {wc['w'].sum():,.1f}; on survivors "
        f"{wc.loc[surv_mask, 'w'].sum():,.1f} "
        f"({100*wc.loc[surv_mask,'w'].sum()/wc['w'].sum():.1f}%)")
    an_t = set(an.loc[an.anchor_source == "tstar", "mblogid"])
    unmatched_t = an_t - set(mp["mblogid_t"])
    say(f"treated with anchor but never matched: {len(unmatched_t)}")
    wt = w[w["mblogid"].isin(unmatched_t)]
    say(f"  of these, present in weights.parquet: {len(wt)} "
        f"(w values: {sorted(wt['w'].unique().tolist())[:5]})")

    with open(os.path.join(OUTDIR, "drop_and_censoring.txt"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    say("\nWROTE ->", os.path.join(OUTDIR, "drop_and_censoring.txt"))


if __name__ == "__main__":
    main()
