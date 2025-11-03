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
    y = threads.set_index("mblogid").loc[model_input["mblogid"], "treated"].astype(bool).values

    # Topic embedding via TF-IDF + SVD
    tfidf = TfidfVectorizer(max_features=cfg["tfidf_max_features"], ngram_range=(1,2), min_df=2)
    svd = TruncatedSVD(n_components=cfg["svd_components"], random_state=cfg["seed"])

    preproc = ColumnTransformer([
        ("passthrough", "passthrough", ["post_len","post_has_question","post_has_hashtag","pic_num","hour_of_day","day_of_week"]),
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
    topic_scores = np.zeros((len(model_input), svd.n_components))
    try:
        tfidf_svd_fitted = model.named_steps["pre"].named_transformers_.get("tfidf_svd")
        if tfidf_svd_fitted is not None:
            transformed = tfidf_svd_fitted.transform(model_input["content"].values)
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            topic_scores = np.asarray(transformed)
    except Exception as exc:
        print(f"Warning: topic embedding transform failed ({exc}); falling back to zeros.")
        topic_scores = np.zeros((len(model_input), svd.n_components))

    model_input["topic_component"] = topic_scores[:, 0] if topic_scores.shape[1] > 0 else 0.0
    if topic_scores.shape[1] > 1:
        model_input["topic_component_2"] = topic_scores[:, 1]
    else:
        model_input["topic_component_2"] = 0.0

    topic_vector = model_input[["topic_component", "topic_component_2"]].to_numpy()
    vector_magnitude = np.linalg.norm(topic_vector, axis=1)
    unique_topics = pd.Series(vector_magnitude).nunique(dropna=False)
    if unique_topics > 1:
        q = min(10, unique_topics)
        model_input["topic_band"] = pd.qcut(vector_magnitude, q=q, labels=False, duplicates="drop")
    else:
        model_input["topic_band"] = 0
    model_input["hour_of_day_band"] = pd.cut(model_input["hour_of_day"], bins=[-1,5,11,17,23], labels=["night","morning","afternoon","evening"])
    tmeta = model_input[["mblogid","topic_band","hour_of_day_band"]].drop_duplicates()

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
            C = control[(control["topic_band"]==r.topic_band) & (control["hour_of_day_band"]==r.hour_of_day_band)].copy()
            if C.empty:
                continue
            C["dist"] = np.abs(C["logit"] - r.logit)
            C = C[C["dist"] <= cfg["caliper_logit"]].nsmallest(cfg["k_controls"], "dist")
            for rr in C.itertuples(index=False):
                pairs.append({"mblogid_t": r.mblogid, "mblogid_c": rr.mblogid, "dist": rr.dist})
        pairs = pd.DataFrame(pairs)
    save_parquet(pairs, f"{ART}/matched_pairs.parquet")

    # Diagnostics
    diag = pd.DataFrame({"metric":["AUC","n_treated","n_pairs"], "value":[auc, treated.shape[0], pairs.shape[0]]})
    save_parquet(diag, f"{ART}/diagnostics.parquet")
    print("Propensity & matching complete. AUC=%.3f, matched pairs=%d" % (auc, len(pairs)))

    # Balance diagnostics (numerical covariates)
    numeric_features = [col for col in ["post_len","post_has_question","post_has_hashtag","pic_num","hour_of_day","day_of_week","topic_component"] if col in model_input.columns]
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
    outcomes = pd.read_parquet("artifacts/outcomes/outcomes.parquet")
    outcomes = outcomes.set_index("mblogid")
    effect_rows = []
    if not pairs.empty:
        effect_pairs = pairs[
            pairs["mblogid_t"].isin(outcomes.index) & pairs["mblogid_c"].isin(outcomes.index)
        ]
    else:
        effect_pairs = pd.DataFrame()
    if not effect_pairs.empty:
        metrics = [
            "d_R",
            "d_R_with_op",
            "d_BF",
            "d_Gini",
            "d_DCBI_gender",
            "d_DCBI_province",
            "d_prosocial_any",
            "d_prosocial_score",
            "d_Cross_gender_share",
            "d_Cross_province_share",
        ]
        for metric in metrics:
            if metric not in outcomes.columns:
                continue
            treated_vals = outcomes.loc[effect_pairs["mblogid_t"], metric]
            control_vals = outcomes.loc[effect_pairs["mblogid_c"], metric]
            df_metric = pd.DataFrame({"treated": treated_vals.values, "control": control_vals.values}).dropna()
            treated_mean = df_metric["treated"].mean() if not df_metric.empty else np.nan
            control_mean = df_metric["control"].mean() if not df_metric.empty else np.nan
            if df_metric.empty:
                diff_mean = np.nan
                se = np.nan
                n_pairs = 0
            else:
                diff = df_metric["treated"] - df_metric["control"]
                diff_mean = diff.mean()
                se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
                n_pairs = len(diff)
            effect_rows.append({
                "metric": metric,
                "treated_mean": treated_mean,
                "control_mean": control_mean,
                "diff_mean": diff_mean,
                "diff_se": se,
                "n_pairs": n_pairs
            })
    matched_effects = pd.DataFrame(effect_rows)
    save_parquet(matched_effects, f"{ART}/matched_effects.parquet")

if __name__ == "__main__":
    run()
