import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..models.style import extract_style_features
from ..utils.io_utils import save_parquet, ensure_dir
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

ART = "artifacts/style"

def _detect_agent(user_obj, cfg) -> bool:
    if not isinstance(user_obj, dict):
        return False
    uid = user_obj.get("_id")
    nickname = user_obj.get("nick_name", "") or ""
    verified = bool(user_obj.get("verified", False))
    if cfg.get("agent_ids") and uid in cfg["agent_ids"]:
        return True
    if cfg.get("agent_verified_required", True) and not verified:
        return False
    for variant in cfg.get("agent_nickname_variants", []):
        if variant and variant in nickname:
            return True
    return False


def run():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    comments = pd.read_parquet("artifacts/ingested/comments.parquet")
    tstars = pd.read_parquet("artifacts/threads/tstars.parquet")
    outcomes = pd.read_parquet("artifacts/outcomes/outcomes.parquet")
    if "d_DCBI_gender" not in outcomes.columns and {"pre_DCBI_gender", "post_DCBI_gender"}.issubset(outcomes.columns):
        outcomes = outcomes.copy()
        outcomes["d_DCBI_gender"] = outcomes["post_DCBI_gender"] - outcomes["pre_DCBI_gender"]

    treated_threads = tstars[tstars["tstar"].notna()]["mblogid"].tolist()
    if not treated_threads:
        ensure_dir(ART)
        save_parquet(pd.DataFrame(), f"{ART}/style_features.parquet")
        save_parquet(pd.DataFrame(), f"{ART}/feature_importance.parquet")
        print("Style features skipped: no treated threads detected.")
        return

    subset = comments[comments["root_post_mblogid"].isin(treated_threads)].copy()
    subset["is_agent"] = subset["comment_user"].apply(lambda u: _detect_agent(u, cfg))
    subset = subset[subset["is_agent"]]
    if subset.empty:
        ensure_dir(ART)
        save_parquet(pd.DataFrame(), f"{ART}/style_features.parquet")
        save_parquet(pd.DataFrame(), f"{ART}/feature_importance.parquet")
        print("Style features skipped: treated threads have no detected agent replies.")
        return

    subset = subset.sort_values("created_at")
    firsts = subset.groupby("root_post_mblogid").head(1).copy()

    # Extract style features
    feats = []
    for r in tqdm(firsts.itertuples(index=False), total=len(firsts), desc="Extracting style features"):
        s = extract_style_features(r.content or "", cfg["empathy_markers"], cfg["politeness_markers"], cfg["hedges"])
        s["mblogid"] = r.root_post_mblogid
        feats.append(s)
    F = pd.DataFrame(feats)

    # Outcome: d_DCBI_gender (post-pre) as primary bridging
    Y = outcomes[["mblogid","d_DCBI_gender"]].dropna()
    DF = Y.merge(F, on="mblogid", how="inner")

    if DF.empty:
        ensure_dir(ART)
        save_parquet(F, f"{ART}/style_features.parquet")
        save_parquet(pd.DataFrame(), f"{ART}/feature_importance.parquet")
        print("Style features extracted but no overlapping outcomes; skipping HTE model.")
        return

    X = DF.drop(columns=["mblogid","d_DCBI_gender"])
    y = DF["d_DCBI_gender"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=cfg["seed"])
    rf = RandomForestRegressor(random_state=cfg["seed"])
    rf.fit(X_train, y_train)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).rename("importance").reset_index().rename(columns={"index":"feature"})

    ensure_dir(ART)
    save_parquet(F, f"{ART}/style_features.parquet")
    save_parquet(imp, f"{ART}/feature_importance.parquet")
    print(f"Style features extracted for {len(F):,} treated threads. RandomForest proxy HTE importance saved.")

if __name__ == "__main__":
    run()
