import yaml
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from ..utils.io_utils import save_parquet, ensure_dir
from ..utils.text_utils import has_question, has_hashtag, char_len

ART = "artifacts/matching"

def run():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    posts = pd.read_parquet("artifacts/ingested/posts.parquet")
    comments = pd.read_parquet("artifacts/ingested/comments.parquet")
    threads = pd.read_parquet("artifacts/ingested/threads.parquet")
    tstars = pd.read_parquet("artifacts/threads/tstars.parquet")
    try:
        thread_meta = pd.read_parquet("artifacts/threads/thread_meta.parquet")
    except Exception:
        thread_meta = pd.DataFrame()

    # Optional dev/debug: restrict to the same thread subset produced by step02/step03
    # This avoids training the propensity model on the full ingested corpus when you
    # are iterating with debug_max_threads.
    max_threads = int(cfg.get("debug_max_threads", 0) or 0)
    if max_threads and not thread_meta.empty and "mblogid" in thread_meta.columns:
        keep_ids = (
            thread_meta["mblogid"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(max_threads)
            .tolist()
        )
        if keep_ids:
            posts = posts[posts["mblogid"].astype(str).isin(keep_ids)].copy()
            threads = threads[threads["mblogid"].astype(str).isin(keep_ids)].copy()
            # comments are not used by the propensity model, but keep consistent anyway
            if "root_post_mblogid" in comments.columns:
                comments = comments[comments["root_post_mblogid"].astype(str).isin(keep_ids)].copy()

    # Label treated if t* present
    treated_ids = set(tstars.loc[tstars["tstar"].notna(), "mblogid"])
    threads["treated"] = threads["mblogid"].isin(treated_ids)

    # Pre-treatment covariates X
    def features_df(df):
        X = pd.DataFrame({
            "post_len": df["content"].apply(char_len),
            "post_has_question": df["content"].apply(lambda t: int(has_question(t or ""))),
            "post_has_hashtag": df["content"].apply(lambda t: int(has_hashtag(t or ""))),
            "pic_num": df["pic_num"].fillna(0).astype(int),
            "hour_of_day": df["created_at"].dt.hour,
            "day_of_week": df["created_at"].dt.dayofweek,
            "source": df["source"].astype(str).fillna(""),
        })
        return X

    X_base = features_df(posts)
    model_input = posts.copy()
    for col in X_base.columns:
        model_input[col] = X_base[col]
    model_input["content"] = model_input["content"].fillna("")

    # Optional: add early engagement proxy from step02 (pre-treatment, based on first Δ minutes)
    if not thread_meta.empty and "mblogid" in thread_meta.columns and "early_human_comments" in thread_meta.columns:
        early = thread_meta[["mblogid", "early_human_comments"]].drop_duplicates("mblogid")
        model_input = model_input.merge(early, on="mblogid", how="left")
        model_input["early_human_comments"] = (
            pd.to_numeric(model_input["early_human_comments"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    else:
        model_input["early_human_comments"] = 0

    # Coarse "lifecycle stage" proxy for matching: early engagement band
    # (helps align treated threads that get early vs late agent replies)
    model_input["early_engagement_band"] = pd.cut(
        pd.to_numeric(model_input["early_human_comments"], errors="coerce").fillna(0),
        bins=[-1, 0, 2, 5, 10, np.inf],
        labels=["0", "1-2", "3-5", "6-10", "11+"],
    ).astype(str)

    y = threads.set_index("mblogid").loc[model_input["mblogid"], "treated"].astype(bool).values

    # Topic embedding via TF-IDF + SVD
    tfidf = TfidfVectorizer(max_features=cfg["tfidf_max_features"], ngram_range=(1,2), min_df=2)
    svd = TruncatedSVD(n_components=cfg["svd_components"], random_state=cfg["seed"])

    preproc = ColumnTransformer([
        ("passthrough", "passthrough", ["post_len","post_has_question","post_has_hashtag","pic_num","hour_of_day","day_of_week","early_human_comments"]),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), ["source"]),
        ("tfidf_svd", Pipeline([("tfidf", tfidf), ("svd", svd)]), "content"),
    ])

    model = Pipeline([
        ("pre", preproc),
        ("gbdt", GradientBoostingClassifier(random_state=cfg["seed"]))
    ])

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        ps = np.full(len(model_input), float(unique_classes[0]) if unique_classes.size else 0.0)
        auc = float("nan")
    else:
        print(f"Training propensity model on {len(model_input):,} posts...")
        model.fit(model_input, y)
        print("Propensity model fit complete.")
        ps = model.predict_proba(model_input)[:,1]
        auc = roc_auc_score(y, ps)

    # Save per-thread propensity (average over same post index)
    prop = pd.DataFrame({"mblogid": model_input["mblogid"], "propensity": ps})
    prop = prop.groupby("mblogid", as_index=False)["propensity"].mean()
    ensure_dir(ART)
    save_parquet(prop, f"{ART}/propensity.parquet")

    # Matching: nearest neighbors on logit with caliper; exact on hour band and topic band
    # For simplicity here, we will do coarse exact matches on hour_of_day band and SVD topic band
    try:
        topic_scores = svd.transform(tfidf.transform(model_input["content"]))[:,0]
    except Exception:
        topic_scores = np.zeros(len(model_input))
    model_input["topic_component"] = topic_scores
    unique_topics = pd.Series(model_input["topic_component"]).nunique(dropna=False)
    if unique_topics > 1:
        q = min(10, unique_topics)
        model_input["topic_band"] = pd.qcut(model_input["topic_component"], q=q, labels=False, duplicates="drop")
    else:
        model_input["topic_band"] = 0
    model_input["hour_of_day_band"] = pd.cut(model_input["hour_of_day"], bins=[-1,5,11,17,23], labels=["night","morning","afternoon","evening"])
    # Exact/coarsened match fields (configurable)
    match_fields = cfg.get("exact_match_fields") or ["hour_of_day_band", "topic_band"]
    match_fields = [str(f) for f in match_fields]
    # NEW: lifecycle-stage proxy (always include; harmless but improves comparability)
    if "early_engagement_band" in model_input.columns and "early_engagement_band" not in match_fields:
        match_fields = match_fields + ["early_engagement_band"]
    # Keep backward compatibility if some fields are missing
    match_fields = [f for f in match_fields if f in model_input.columns]
    tmeta = model_input[["mblogid"] + match_fields].drop_duplicates()

    m = threads.merge(prop, on="mblogid").merge(tmeta, on="mblogid")
    prop_clip = m["propensity"].clip(1e-6, 1-1e-6)
    m["logit"] = np.log(prop_clip / (1 - prop_clip))

    treated = m[m["treated"]]
    control = m[~m["treated"]]

    pairs = []
    if len(unique_classes) < 2 or treated.empty or control.empty:
        pairs = pd.DataFrame(columns=["mblogid_t","mblogid_c","dist"])
    else:
        for r in tqdm(treated.itertuples(index=False), total=len(treated), desc="Matching treated threads"):
            # strict matched anchors only: controls must be from the matched pool
            # and also align on coarse lifecycle stage + time/topic bands
            C = control
            for f in match_fields:
                C = C[C[f] == getattr(r, f)]
            C = C.copy()
            if C.empty:
                continue
            C["dist"] = np.abs(C["logit"] - r.logit)
            C = C[C["dist"] <= cfg["caliper_logit"]].nsmallest(cfg["k_controls"], "dist")
            for rr in C.itertuples(index=False):
                pairs.append({"mblogid_t": r.mblogid, "mblogid_c": rr.mblogid, "dist": rr.dist})
        pairs = pd.DataFrame(pairs)
    save_parquet(pairs, f"{ART}/matched_pairs.parquet")

    # NEW: Valid pairs are the only analysis population downstream.
    # Keep file name stable for the new implementation.
    valid_pairs = pairs.dropna(subset=["mblogid_t", "mblogid_c"]).copy()
    if not valid_pairs.empty:
        valid_pairs["mblogid_t"] = valid_pairs["mblogid_t"].astype(str)
        valid_pairs["mblogid_c"] = valid_pairs["mblogid_c"].astype(str)
    save_parquet(valid_pairs, f"{ART}/valid_pairs.parquet")

    # Diagnostics
    diag = pd.DataFrame(
        {
            "metric": ["AUC", "n_treated", "n_pairs", "n_valid_pairs", "n_controls_matched"],
            "value": [
                auc,
                treated.shape[0],
                pairs.shape[0],
                valid_pairs.shape[0],
                valid_pairs["mblogid_c"].nunique() if not valid_pairs.empty else 0,
            ],
        }
    )
    save_parquet(diag, f"{ART}/diagnostics.parquet")
    print("Propensity & matching complete. AUC=%.3f, matched pairs=%d" % (auc, len(pairs)))

    # Balance diagnostics (numerical covariates)
    numeric_features = [
        col
        for col in [
            "post_len",
            "post_has_question",
            "post_has_hashtag",
            "pic_num",
            "hour_of_day",
            "day_of_week",
            "early_human_comments",
            "topic_component",
        ]
        if col in model_input.columns
    ]
    features = model_input[["mblogid"] + numeric_features].drop_duplicates("mblogid").set_index("mblogid")

    def standardized_diff(t_df, c_df):
        rows = []
        for col in numeric_features:
            t = pd.to_numeric(t_df[col], errors="coerce")
            c = pd.to_numeric(c_df[col], errors="coerce")
            t = t.dropna()
            c = c.dropna()
            if t.empty or c.empty:
                rows.append({"feature": col, "treated_mean": np.nan, "control_mean": np.nan, "std_diff": np.nan})
                continue
            diff = t.mean() - c.mean()
            pooled = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2) if (t.var(ddof=1) + c.var(ddof=1)) > 0 else np.nan
            rows.append({
                "feature": col,
                "treated_mean": t.mean(),
                "control_mean": c.mean(),
                "std_diff": diff / pooled if pooled and not np.isnan(pooled) else np.nan
            })
        return pd.DataFrame(rows)

    treated_ids_full = threads.loc[threads["treated"], "mblogid"]
    control_ids_full = threads.loc[~threads["treated"], "mblogid"]
    before = standardized_diff(features.loc[treated_ids_full], features.loc[control_ids_full])
    before = before.rename(columns={
        "treated_mean": "treated_mean_before",
        "control_mean": "control_mean_before",
        "std_diff": "std_diff_before"
    })

    balance_pairs = pairs[
        pairs["mblogid_t"].isin(features.index) & pairs["mblogid_c"].isin(features.index)
    ] if not pairs.empty else pd.DataFrame()
    if not balance_pairs.empty and not features.empty:
        treated_matched = features.loc[balance_pairs["mblogid_t"]].reset_index(drop=True)
        control_matched = features.loc[balance_pairs["mblogid_c"]].reset_index(drop=True)
        after = standardized_diff(treated_matched, control_matched)
        after = after.rename(columns={
            "treated_mean": "treated_mean_after",
            "control_mean": "control_mean_after",
            "std_diff": "std_diff_after"
        })
        balance = before.merge(after, on="feature", how="outer")
    else:
        balance = before
    save_parquet(balance, f"{ART}/balance_table.parquet")

    # Matched effects on outcomes
    try:
        outcomes = pd.read_parquet("artifacts/outcomes/outcomes.parquet").set_index("mblogid")
    except Exception:
        outcomes = pd.DataFrame()

    effect_rows = []
    if outcomes is None or outcomes.empty:
        # Allow running matching before outcomes (needed for control pseudo-anchors)
        matched_effects = pd.DataFrame()
        save_parquet(matched_effects, f"{ART}/matched_effects.parquet")
        print("Matched effects skipped: outcomes.parquet not found yet. Run step03 then rerun step04 if needed.")
        return

    if not valid_pairs.empty:
        effect_pairs = valid_pairs[
            valid_pairs["mblogid_t"].isin(outcomes.index) & valid_pairs["mblogid_c"].isin(outcomes.index)
        ]
    else:
        effect_pairs = pd.DataFrame()

    if not effect_pairs.empty:
        metrics = [
            # Sample A (formation): post-only levels on time window
            "post_time_R_with_op",
            "post_time_BF",
            "post_time_Gini",
            "post_time_Assort_province",
            "post_time_DCBI_province",
            "post_time_Stance_Divergence",
            "post_time_Stance_Agonism",
            # Sample B (rewiring): incumbent-only time-window deltas
            "d_time_inc_R_with_op",
            "d_time_inc_BF",
            "d_time_inc_Gini",
            "d_time_inc_Assort_province",
            "d_time_inc_DCBI_province",
            "d_time_inc_Stance_Divergence",
            "d_time_inc_Stance_Agonism",
            # Optional tone deltas (if present)
            "d_prosocial_any",
            "d_prosocial_score",
        ]
        for metric in metrics:
            if metric not in outcomes.columns:
                continue
            treated_vals = outcomes.loc[effect_pairs["mblogid_t"], metric]
            control_vals = outcomes.loc[effect_pairs["mblogid_c"], metric]
            df_metric = pd.DataFrame({"treated": treated_vals.values, "control": control_vals.values}).dropna()
            if df_metric.empty:
                continue
            diff = df_metric["treated"] - df_metric["control"]
            se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
            effect_rows.append({
                "metric": metric,
                "treated_mean": df_metric["treated"].mean(),
                "control_mean": df_metric["control"].mean(),
                "diff_mean": diff.mean(),
                "diff_se": se,
                "n_pairs": len(diff)
            })

    matched_effects = pd.DataFrame(effect_rows)
    save_parquet(matched_effects, f"{ART}/matched_effects.parquet")

if __name__ == "__main__":
    run()
