"""
a_topic.py -- TOPIC HETEROGENEITY of the formation effect (AE-4, R3-add-2).

Q: Is the focal-shift (formation effect) stronger in information-seeking topics
   (tech/health, politics/news) than expressive ones (entertainment/life)?
   Tests the satiation mechanism.

Design:
  - Transparent hashtag+keyword topic classifier over posts.parquet content.
  - Buckets: entertainment, life_emotion, politics_news, tech_health, sports, other.
  - Restrict to sample_group=='early' (Sample A).
  - Merge topic onto TREATED threads.
  - For each bucket: subsample = (treated in bucket) + (their matched controls
    via matched_pairs) ; estimate matched_dr ATT on:
       post_time_R_with_op (reciprocity), post_time_BF, post_time_Assort_province.

NOTE: This is EXPLORATORY heterogeneity built from a transparent keyword/hashtag
classifier -- NOT a validated NLP model. Bucket membership is coarse and the
'other' residual is large by construction. Interpret directionally.
"""
import os
import re
import json
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
import pandas as pd
import revlib as R

ART = r"D:\research  tasks\TSC revision\artifacts"
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "results", "topic")
OUTDIR = os.path.abspath(OUTDIR)
os.makedirs(OUTDIR, exist_ok=True)

OUTCOMES = ["post_time_R_with_op", "post_time_BF", "post_time_Assort_province"]

# ------------------------------------------------------------ topic keywords
# Chinese keyword lexicon. Order = classification priority (first hit wins,
# except entertainment/life which are lower priority than the information
# buckets so that a clearly-newsy post is not stolen by a generic life word).
KEYWORDS = {
    "politics_news": ["新闻", "政策", "政府", "国际", "事件", "时政", "官方",
                       "通报", "会议", "领导", "外交", "两会", "疫情"],
    "tech_health":   ["科技", "健康", "医", "疫苗", "AI", "人工智能", "手机",
                       "电脑", "互联网", "数码", "芯片", "科学", "研究", "疾病",
                       "养生", "医院", "药"],
    "sports":        ["体育", "足球", "篮球", "比赛", "奥运", "世界杯", "球赛",
                       "运动员", "冠军", "联赛", "球队", "NBA", "CBA"],
    "entertainment": ["明星", "电影", "综艺", "娱乐", "剧", "演员", "歌手",
                       "追星", "偶像", "票房", "电视剧", "音乐", "演唱会", "网红"],
    "life_emotion":  ["情感", "生活", "家庭", "爱情", "心情", "感情", "婚姻",
                       "美食", "旅行", "日常", "宝宝", "孩子", "分享", "喜欢"],
}
# Classification priority: information-seeking first, then expressive.
PRIORITY = ["politics_news", "tech_health", "sports", "entertainment", "life_emotion"]

HASHTAG_RE = re.compile(r"#([^#]+)#")


def classify(text):
    """Return topic bucket for one post. Hashtag text is weighted the same as
    body text (both scanned). First bucket (in PRIORITY order) with any keyword
    hit wins; else 'other'."""
    if not isinstance(text, str) or not text:
        return "other"
    # scan whole content (hashtags are inside content as #...# already)
    for bucket in PRIORITY:
        for kw in KEYWORDS[bucket]:
            if kw in text:
                return bucket
    return "other"


def main():
    # ---- load core (Sample A / early) with outcomes
    df = R.load_core(outcome_cols=OUTCOMES)
    df = df[df["sample_group"] == "early"].copy()

    # ---- sanity check FIRST: pooled early reciprocity matched_dr att ~ -0.167
    san = R.matched_dr_att(df, "post_time_R_with_op", seed=0)
    san_att = float(san["att"])
    print(f"[SANITY] pooled early reciprocity matched_dr att = {san_att:.4f} "
          f"(ref ~ -0.167), n_t={san['n_treated']} n_c={san['n_control']}")
    sanity_pass = abs(san_att - (-0.167)) < 0.02
    if not sanity_pass:
        print("[SANITY] FAILED -- att not within 0.02 of -0.167. STOPPING.")
        # still save what we have for diagnosis
        with open(os.path.join(OUTDIR, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"sanity_pass": False, "pooled_reciprocity_att": san_att}, f, indent=2)
        raise SystemExit(1)
    print("[SANITY] PASSED.")

    # ---- classify topics on posts.parquet (content) -- only need treated ids,
    # but classify all so we can report bucket sizes over the analyzable posts.
    posts = pd.read_parquet(os.path.join(ART, "ingested", "posts.parquet"),
                            columns=["mblogid", "content"])
    posts["topic"] = posts["content"].map(classify)

    # treated threads in early
    treated = df[df["T"] == 1][["mblogid"]].copy()
    treated = treated.merge(posts[["mblogid", "topic"]], on="mblogid", how="left")
    treated["topic"] = treated["topic"].fillna("other")

    # bucket sizes over TREATED early threads (the population we actually split)
    bucket_sizes = treated["topic"].value_counts().to_dict()
    print("[BUCKETS] treated-early sizes:", bucket_sizes)

    # example keywords per bucket for auditability (5 each)
    example_kw = {b: KEYWORDS.get(b, [])[:5] for b in PRIORITY}
    example_kw["other"] = ["(no keyword hit)"]

    # matched pairs
    mp = pd.read_parquet(os.path.join(ART, "matching", "matched_pairs.parquet"))

    buckets = PRIORITY + ["other"]
    all_rows = []
    recip_findings = {}
    for b in buckets:
        tids = set(treated.loc[treated["topic"] == b, "mblogid"])
        if not tids:
            continue
        cids = set(mp.loc[mp["mblogid_t"].isin(tids), "mblogid_c"])
        sub = df[((df["T"] == 1) & (df["mblogid"].isin(tids))) |
                 ((df["T"] == 0) & (df["mblogid"].isin(cids)))].copy()
        n_t = int((sub["T"] == 1).sum())
        n_c = int((sub["T"] == 0).sum())
        for oc in OUTCOMES:
            res = R.matched_dr_att(sub, oc, seed=0)
            row = dict(topic=b, outcome=oc, estimator="matched_dr",
                       att=res["att"], se=res["se"], p=res["p"],
                       n_treated=res["n_treated"], n_control=res["n_control"])
            all_rows.append(row)
            if oc == "post_time_R_with_op":
                recip_findings[b] = row
        print(f"[BUCKET {b}] n_t(bucket)={n_t} n_c={n_c} "
              f"recip att={recip_findings.get(b,{}).get('att')}")

    results = pd.DataFrame(all_rows)

    # ---- save
    results.to_parquet(os.path.join(OUTDIR, "topic_att.parquet"), index=False)
    results.to_csv(os.path.join(OUTDIR, "topic_att.csv"), index=False)

    bs_df = pd.DataFrame([{"topic": k, "n_treated_early": v}
                          for k, v in bucket_sizes.items()])
    bs_df.to_csv(os.path.join(OUTDIR, "bucket_sizes.csv"), index=False)

    summary = {
        "sanity_pass": True,
        "pooled_early_reciprocity_matched_dr_att": san_att,
        "sanity_ref": -0.167,
        "outcomes": OUTCOMES,
        "bucket_sizes_treated_early": bucket_sizes,
        "plan_sizes_note": {"entertainment": 30000, "life_emotion": 27000,
                             "politics_news": 10000, "tech_health": 7000,
                             "sports": 6000},
        "example_keywords_per_bucket": example_kw,
        "reciprocity_att_by_topic": {
            b: {"att": recip_findings[b]["att"], "se": recip_findings[b]["se"],
                "p": recip_findings[b]["p"], "n_treated": recip_findings[b]["n_treated"]}
            for b in recip_findings},
        "caveat": ("Exploratory heterogeneity from a transparent keyword/hashtag "
                   "classifier, NOT a validated NLP model. Coarse buckets; large "
                   "'other' residual. Interpret directionally."),
    }
    with open(os.path.join(OUTDIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # SUMMARY.md
    lines = ["# Topic Heterogeneity of the Formation Effect (AE-4, R3-add-2)\n",
             f"Sanity (pooled early reciprocity matched_dr att): **{san_att:.4f}** "
             f"(ref ~ -0.167) -> PASS\n",
             "\n## Bucket sizes (treated, early)\n",
             "| topic | n_treated_early | plan |", "|---|---|---|"]
    plan = summary["plan_sizes_note"]
    for b in buckets:
        lines.append(f"| {b} | {bucket_sizes.get(b,0)} | {plan.get(b,'-')} |")
    lines.append("\n## Reciprocity (post_time_R_with_op) matched_dr ATT by topic\n")
    lines.append("| topic | att | se | p | n_treated |")
    lines.append("|---|---|---|---|---|")
    for b in buckets:
        if b in recip_findings:
            r = recip_findings[b]
            lines.append(f"| {b} | {r['att']:.4f} | {r['se']:.4f} | "
                         f"{r['p']:.3g} | {r['n_treated']} |")
    lines.append("\n## All outcomes (matched_dr ATT)\n")
    lines.append("| topic | outcome | att | se | p | n_t | n_c |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in results.iterrows():
        lines.append(f"| {r['topic']} | {r['outcome']} | {r['att']:.4f} | "
                     f"{r['se']:.4f} | {r['p']:.3g} | {int(r['n_treated'])} | "
                     f"{int(r['n_control'])} |")
    lines.append("\n## Interpretation\n")
    lines.append("Satiation mechanism predicts a *stronger* (more negative) "
                 "reciprocity focal-shift in information-seeking buckets "
                 "(tech_health, politics_news) than expressive ones "
                 "(entertainment, life_emotion). See table above.\n")
    lines.append("\n**Caveat:** " + summary["caveat"] + "\n")
    with open(os.path.join(OUTDIR, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[DONE] saved to", OUTDIR)


if __name__ == "__main__":
    main()
