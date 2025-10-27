import os, yaml, pandas as pd, numpy as np
from ..metrics.network_metrics import reciprocity_stats, branching_factor, equality_of_voice, assortativity, dc_bi_analytic
from ..utils.io_utils import save_parquet, ensure_dir
from ..models.prosocial import ProsocialAnalyzer

ART = "artifacts/outcomes"

def run():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    edges = pd.read_parquet("artifacts/threads/edges.parquet")
    tstars = pd.read_parquet("artifacts/threads/tstars.parquet")
    meta = pd.read_parquet("artifacts/threads/thread_meta.parquet")
    users = pd.read_parquet("artifacts/ingested/users.parquet")
    posts = pd.read_parquet("artifacts/ingested/posts.parquet")
    comments = pd.read_parquet("artifacts/ingested/comments.parquet")

    # Build user attribute table
    ua = users[["_id","verified","followers_count","gender","location","ip_location","mbrank","mbtype","created_at","statuses_count","sunshine_credit"]].copy()
    ua.rename(columns={"_id":"user_id","ip_location":"province"}, inplace=True)
    ua["gender"] = ua["gender"].fillna("unk").replace({"": "unk"})
    ua["province"] = ua["province"].fillna("unk").replace({"": "unk"})

    # Prosocial analyser (lexicon + classifier if available)
    prosocial = ProsocialAnalyzer.from_config(cfg, comments)

    # Pre/post windows per thread
    time_win = cfg["time_window_minutes"]
    tstars_map = dict(zip(tstars.mblogid, tstars.tstar))

    rows = []
    for mblogid, T_edges in edges.groupby("root_post_mblogid"):
        tstar = tstars_map.get(mblogid, pd.NaT)
        op_id = meta.loc[meta.mblogid==mblogid, "op_id"].iloc[0] if (meta.mblogid==mblogid).any() else None
        T_comments = comments[comments["root_post_mblogid"]==mblogid].copy().sort_values("created_at")

        if not pd.isna(tstar):
            pre_mask = (T_edges["created_at"] < tstar) & (T_edges["created_at"] >= tstar - pd.Timedelta(minutes=time_win))
            post_mask = (T_edges["created_at"] >= tstar) & (T_edges["created_at"] <= tstar + pd.Timedelta(minutes=time_win))
        else:
            pre_mask = T_edges["created_at"] < T_edges["created_at"].min()
            post_mask = T_edges["created_at"] >= T_edges["created_at"].min()

        def metrics_for(df_edges):
            if df_edges.empty:
                return {"R": np.nan, "R_weighted": np.nan, "BF": np.nan, "BF_proxy": np.nan, "Gini": np.nan, "Assort_gender": np.nan, "Assort_province": np.nan, "DCBI_gender": np.nan, "DCBI_province": np.nan}
            R = reciprocity_stats(df_edges, op_id=op_id, exclude_op=cfg["exclude_op_in_reciprocity_main"])
            BF = branching_factor(df_edges, op_id=op_id)
            G = equality_of_voice(df_edges)
            nodes = pd.unique(df_edges[['u','v']].values.ravel('K'))
            uattr = ua[ua["user_id"].isin(nodes)][["user_id","gender","province","verified"]].copy()
            A_gender = assortativity(df_edges, uattr[["user_id","gender"]], "gender")
            A_prov = assortativity(df_edges, uattr[["user_id","province"]], "province")
            DCBI_gender, _ = dc_bi_analytic(df_edges, uattr[["user_id","gender"]], "gender")
            DCBI_prov, _ = dc_bi_analytic(df_edges, uattr[["user_id","province"]], "province")
            out = {}
            out.update(R)
            out.update(BF)
            out.update(G)
            out["Assort_gender"] = A_gender
            out["Assort_province"] = A_prov
            out["DCBI_gender"] = DCBI_gender
            out["DCBI_province"] = DCBI_prov
            return out

        all_m = metrics_for(T_edges)
        pre_m = metrics_for(T_edges[pre_mask])
        post_m = metrics_for(T_edges[post_mask])

        if not pd.isna(tstar):
            pre_c = T_comments[(T_comments["created_at"] < tstar) & (T_comments["created_at"] >= tstar - pd.Timedelta(minutes=time_win))]
            post_c = T_comments[(T_comments["created_at"] >= tstar) & (T_comments["created_at"] <= tstar + pd.Timedelta(minutes=time_win))]
        else:
            pre_c = T_comments.iloc[0:0]
            post_c = T_comments

        P_all = prosocial.aggregate(T_comments, "content")
        P_pre = prosocial.aggregate(pre_c, "content")
        P_post = prosocial.aggregate(post_c, "content")

        row = {
            "mblogid": mblogid,
            "tstar": tstar,
            **{f"all_{k}":v for k,v in all_m.items()},
            **{f"pre_{k}":v for k,v in pre_m.items()},
            **{f"post_{k}":v for k,v in post_m.items()},
            **{f"all_{k}":v for k,v in P_all.items()},
            **{f"pre_{k}":v for k,v in P_pre.items()},
            **{f"post_{k}":v for k,v in P_post.items()},
            "n_edges": len(T_edges)
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    # Add post-pre deltas for any metric with both columns
    for col in out.columns:
        if col.startswith("post_"):
            metric = col[5:]
            pre_col = f"pre_{metric}"
            if pre_col in out.columns:
                out[f"d_{metric}"] = out[col] - out[pre_col]

    ensure_dir(ART)
    save_parquet(out, f"{ART}/outcomes.parquet")
    print("Outcomes computed and saved.")

if __name__ == "__main__":
    run()
