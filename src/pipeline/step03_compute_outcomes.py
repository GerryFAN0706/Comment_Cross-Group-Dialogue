import os, yaml, pandas as pd, numpy as np
from ..metrics.network_metrics import reciprocity_stats, branching_factor, equality_of_voice, assortativity, dc_bi_analytic
from ..utils.io_utils import save_parquet, ensure_dir
from ..utils.anchor_utils import build_control_anchor_map, median_timestamp
from ..models.prosocial import ProsocialAnalyzer

ART = "artifacts/outcomes"

# Simple stance/sentiment lexicon for "low-fidelity" ideological proxy
# (Agreement/Positive vs Disagreement/Negative)
STANCE_LEXICON = {
    "pos": ["支持", "赞", "同意", "对", "不错", "喜欢", "感谢", "加油", "好", "确实", "有理", "顶", "必须", "正确", "合理"],
    "neg": ["反对", "瞎说", "不对", "乱讲", "差", "讨厌", "恶心", "滚", "垃圾", "无语", "并不是", "不觉得", "错", "傻", "有病"],
}

def _compute_sentiment(text: str, pos_words=None, neg_words=None) -> int:
    """Returns 1 (pos), -1 (neg), or 0 (neutral/mixed)"""
    if not isinstance(text, str):
        return 0
    pos_words = list(pos_words) if pos_words is not None else STANCE_LEXICON["pos"]
    neg_words = list(neg_words) if neg_words is not None else STANCE_LEXICON["neg"]
    # low-fidelity proxy: count distinct lexicon hits (not tokenized)
    pos_hits = sum(1 for w in pos_words if w and w in text)
    neg_hits = sum(1 for w in neg_words if w and w in text)
    if pos_hits > neg_hits:
        return 1
    if neg_hits > pos_hits:
        return -1
    return 0

def run():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    edges = pd.read_parquet("artifacts/threads/edges.parquet")
    tstars = pd.read_parquet("artifacts/threads/tstars.parquet")
    meta = pd.read_parquet("artifacts/threads/thread_meta.parquet")
    users = pd.read_parquet("artifacts/ingested/users.parquet")
    posts = pd.read_parquet("artifacts/ingested/posts.parquet")
    comments = pd.read_parquet("artifacts/ingested/comments.parquet")

    # Optional: matched pairs used to assign pseudo anchor times for controls
    try:
        # Prefer the new strict analysis population if available
        pairs = pd.read_parquet("artifacts/matching/valid_pairs.parquet")
    except Exception:
        try:
            pairs = pd.read_parquet("artifacts/matching/matched_pairs.parquet")
        except Exception:
            pairs = pd.DataFrame()

    # Build user attribute table
    ua = users[["_id","verified","followers_count","gender","location","ip_location","mbrank","mbtype","created_at","statuses_count","sunshine_credit"]].copy()
    ua.rename(columns={"_id":"user_id","ip_location":"province"}, inplace=True)
    ua["gender"] = ua["gender"].fillna("unk").replace({"": "unk"})
    ua["province"] = ua["province"].fillna("unk").replace({"": "unk"})

    # Prosocial analyser (lexicon + classifier if available)
    prosocial = ProsocialAnalyzer.from_config(cfg, comments)

    # Window settings
    time_win = int(cfg.get("time_window_minutes", 120))
    K = int(cfg.get("edge_budget_K", 20))
    min_pre = int(cfg.get("edge_budget_min_pre", K))
    min_post = int(cfg.get("edge_budget_min_post", K))
    min_inc_pre = int(cfg.get("incumbent_edge_min_pre", 1))
    min_inc_post = int(cfg.get("incumbent_edge_min_post", 1))
    include_op = bool(cfg.get("incumbent_include_op", True))

    # Sample split thresholds (time-window based; see config.yaml)
    min_pre_h = int(cfg.get("min_pre_human_edges", 2) or 0)
    min_post_h = int(cfg.get("min_post_human_edges", 2) or 0)
    early_max_pre_h = int(cfg.get("early_max_pre_human_edges", max(min_pre_h - 1, 0)) or 0)
    strict_controls_only = bool(cfg.get("strict_matched_controls_only", False))

    # Control anchor strategy label (for traceability in outputs)
    control_strategy = str(cfg.get("control_anchor_strategy", "matched_median") or "matched_median").strip().lower()
    if control_strategy not in {"matched_median", "matched_median_latency"}:
        control_strategy = "matched_median"

    # Anchors
    tstars_map = dict(zip(tstars["mblogid"].astype(str), tstars["tstar"]))
    created_at_map = {}
    if not meta.empty and "mblogid" in meta.columns and "created_at" in meta.columns:
        created_at_map = dict(zip(meta["mblogid"].astype(str), meta["created_at"]))
    control_anchor_map, control_n_matches = build_control_anchor_map(
        pairs,
        tstars_map,
        strategy=str(cfg.get("control_anchor_strategy", "matched_median")),
        created_at_map=created_at_map,
    )

    matched_treated_ids = set(pairs["mblogid_t"].dropna().astype(str)) if (pairs is not None and not pairs.empty and "mblogid_t" in pairs.columns) else set()
    matched_control_ids = set(pairs["mblogid_c"].dropna().astype(str)) if (pairs is not None and not pairs.empty and "mblogid_c" in pairs.columns) else set()

    # Thread -> OP lookup
    op_map = {}
    if not meta.empty and "mblogid" in meta.columns and "op_id" in meta.columns:
        op_map = dict(zip(meta["mblogid"].astype(str), meta["op_id"]))

    # Pre-sort comments and build group index for fast per-thread lookup (avoid O(N^2) filtering)
    comments_sorted = comments.sort_values(["root_post_mblogid", "created_at"]) if not comments.empty else comments
    comments_groups = comments_sorted.groupby("root_post_mblogid", sort=False) if not comments_sorted.empty else None

    rows = []
    if edges.empty:
        out = pd.DataFrame()
    else:
        # Make sure edges are ordered for head/tail slicing
        sort_cols = ["root_post_mblogid", "created_at"]
        if "edge_idx" in edges.columns:
            sort_cols.append("edge_idx")
        edges_sorted = edges.sort_values(sort_cols)

        def _get_thread_comments(mid: str) -> pd.DataFrame:
            if comments_groups is None:
                return comments.iloc[0:0]
            try:
                return comments_groups.get_group(mid)
            except KeyError:
                return comments.iloc[0:0]

        def _human_edges(df_edges: pd.DataFrame) -> pd.DataFrame:
            if df_edges.empty:
                return df_edges
            if "is_human_edge" in df_edges.columns:
                return df_edges[df_edges["is_human_edge"]].copy()
            # Backward-compatible fallback (older artifacts)
            if "is_agent" in df_edges.columns:
                return df_edges[~df_edges["is_agent"]].copy()
            return df_edges.copy()

        def _slice_time_window(df_edges: pd.DataFrame, anchor_time: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
            if df_edges.empty or pd.isna(anchor_time):
                return df_edges.iloc[0:0], df_edges.iloc[0:0]
            window = pd.Timedelta(minutes=time_win)
            pre = df_edges[(df_edges["created_at"] < anchor_time) & (df_edges["created_at"] >= anchor_time - window)]
            post = df_edges[(df_edges["created_at"] >= anchor_time) & (df_edges["created_at"] <= anchor_time + window)]
            return pre, post

        def _slice_edge_budget(df_edges: pd.DataFrame, anchor_time: pd.Timestamp, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
            if df_edges.empty or pd.isna(anchor_time) or k <= 0:
                return df_edges.iloc[0:0], df_edges.iloc[0:0]
            df_edges = df_edges.sort_values("created_at")
            pre = df_edges[df_edges["created_at"] < anchor_time].tail(k)
            post = df_edges[df_edges["created_at"] >= anchor_time].head(k)
            return pre, post

        def _incumbent_only_edges(
            human_df_edges: pd.DataFrame,
            anchor_time: pd.Timestamp,
            op_id: str | None,
        ) -> tuple[pd.DataFrame, set]:
            if human_df_edges.empty or pd.isna(anchor_time):
                return human_df_edges.iloc[0:0], set()
            pre_hist = human_df_edges[human_df_edges["created_at"] < anchor_time]
            incumbents = set(pd.unique(pre_hist[["u", "v"]].values.ravel("K")))
            if include_op and op_id is not None:
                incumbents.add(op_id)
            if not incumbents:
                return human_df_edges.iloc[0:0], set()
            inc_edges = human_df_edges[
                human_df_edges["u"].isin(incumbents) & human_df_edges["v"].isin(incumbents)
            ]
            return inc_edges, incumbents

        def metrics_for(df_edges: pd.DataFrame, op_id: str | None, u_stance: dict | None = None):
            if df_edges is None or df_edges.empty:
                return {
                    "R": np.nan,
                    "R_weighted": np.nan,
                    "R_with_op": np.nan,
                    "R_weighted_with_op": np.nan,
                    "BF": np.nan,
                    "BF_proxy": np.nan,
                    "Gini": np.nan,
                    "Assort_gender": np.nan,
                    "Assort_province": np.nan,
                    "DCBI_gender": np.nan,
                    "DCBI_province": np.nan,
                    "Stance_Divergence": np.nan,
                    "Stance_Agonism": np.nan,
                }
            base = df_edges[["u", "v"]].copy()
            R = reciprocity_stats(
                base,
                op_id=op_id,
                exclude_op=cfg.get("exclude_op_in_reciprocity_main", True),
            )
            R_with_op = reciprocity_stats(base, op_id=op_id, exclude_op=False)
            BF = branching_factor(base, op_id=op_id)
            G = equality_of_voice(base)
            nodes = pd.unique(base[["u", "v"]].values.ravel("K"))
            uattr = ua[ua["user_id"].isin(nodes)][["user_id", "gender", "province", "verified"]].copy()
            A_gender = assortativity(base, uattr[["user_id", "gender"]], "gender")
            A_prov = assortativity(base, uattr[["user_id", "province"]], "province")
            DCBI_gender, _ = dc_bi_analytic(base, uattr[["user_id", "gender"]], "gender")
            DCBI_prov, _ = dc_bi_analytic(base, uattr[["user_id", "province"]], "province")
            
            # Stance metrics
            stance_div = np.nan
            stance_ago = np.nan
            if u_stance:
                # Map u, v to stance scores
                s_u = base["u"].map(u_stance)
                s_v = base["v"].map(u_stance)
                valid = s_u.notna() & s_v.notna()
                if valid.any():
                    diffs = (s_u[valid] - s_v[valid]).abs()
                    stance_div = diffs.mean()
                    # Agonism: fraction of edges with opposite signs (and both non-zero)
                    # i.e. product < 0
                    prods = s_u[valid] * s_v[valid]
                    # We only count agonism if both have expressed a stance (non-zero)
                    non_zeros = (s_u[valid] != 0) & (s_v[valid] != 0)
                    if non_zeros.any():
                        stance_ago = (prods[non_zeros] < 0).mean()
                    else:
                        stance_ago = 0.0

            out = {}
            out.update(R)
            out["R_with_op"] = R_with_op.get("R", np.nan)
            out["R_weighted_with_op"] = R_with_op.get("R_weighted", np.nan)
            out.update(BF)
            out.update(G)
            out["Assort_gender"] = A_gender
            out["Assort_province"] = A_prov
            out["DCBI_gender"] = DCBI_gender
            out["DCBI_province"] = DCBI_prov
            out["Stance_Divergence"] = stance_div
            out["Stance_Agonism"] = stance_ago
            return out

        max_threads = int(cfg.get("debug_max_threads", 0) or 0)
        processed = 0
        for mblogid, T_edges in edges_sorted.groupby("root_post_mblogid", sort=False):
            if max_threads and processed >= max_threads:
                break
            mid = str(mblogid)
            tstar = tstars_map.get(mid, pd.NaT)
            op_id = op_map.get(mid)

            # Strict matched anchors only: drop unmatched controls early to avoid polluting outputs.
            # (Treated threads are kept; downstream analyses will still restrict to matched treated IDs.)
            if strict_controls_only and pd.isna(tstar) and matched_control_ids and (mid not in matched_control_ids):
                continue

            # Anchor time:
            # - treated: t*
            # - matched controls: pseudo anchor from matched treated counterparts (see anchor_utils)
            # - unmatched controls: optional fallback (default: thread median edge time)
            anchor_time = tstar
            anchor_source = "tstar" if not pd.isna(tstar) else "control"
            anchor_n = 0
            if pd.isna(anchor_time):
                anchor_time = control_anchor_map.get(mid, pd.NaT)
                anchor_n = int(control_n_matches.get(mid, 0))
                if not pd.isna(anchor_time):
                    anchor_source = control_strategy
                else:
                    # Fallback (configurable): median edge time within this thread (human edges preferred)
                    fb = str(cfg.get("control_anchor_fallback", "thread_median_edge") or "thread_median_edge").strip().lower()
                    if fb in {"thread_median_edge", "median_edge", "thread_median"}:
                        human_tmp = _human_edges(T_edges)
                        ordered = human_tmp["created_at"].sort_values()
                        anchor_time = median_timestamp(ordered)
                        anchor_source = "thread_median_edge" if not pd.isna(anchor_time) else "missing"
                    else:
                        anchor_time = pd.NaT
                        anchor_source = "missing"

            if strict_controls_only and pd.isna(tstar) and pd.isna(anchor_time):
                # unmatched control under strict mode → drop
                continue

            # Human-only edges (remove agent from both endpoints)
            H_edges = _human_edges(T_edges).sort_values("created_at")

            # Windows:
            # (A) Budget slices (human-only): last K pre, first K post
            pre_budget_edges, post_budget_edges = _slice_edge_budget(H_edges, anchor_time, K)

            # (B) robustness: human-only + time window (±time_win)
            pre_time_edges, post_time_edges = _slice_time_window(H_edges, anchor_time)

            # (C) incumbent set: users observed in pre-history (human-only edges strictly before anchor)
            _, incumbents = _incumbent_only_edges(H_edges, anchor_time, op_id)

            # Incumbent-only edges within (i) budget slices and (ii) time window (NEW main rewiring definition)
            pre_budget_inc_edges = pre_budget_edges[
                pre_budget_edges["u"].isin(incumbents) & pre_budget_edges["v"].isin(incumbents)
            ].copy()
            post_budget_inc_edges = post_budget_edges[
                post_budget_edges["u"].isin(incumbents) & post_budget_edges["v"].isin(incumbents)
            ].copy()
            pre_time_inc_edges = pre_time_edges[
                pre_time_edges["u"].isin(incumbents) & pre_time_edges["v"].isin(incumbents)
            ].copy()
            post_time_inc_edges = post_time_edges[
                post_time_edges["u"].isin(incumbents) & post_time_edges["v"].isin(incumbents)
            ].copy()

            # Comments + stance proxy (used later for prosocial too)
            T_comments = _get_thread_comments(mid)

            # Compute user stance scores for this thread (STRICTLY pre-anchor; human commenters only)
            stance_cfg = cfg.get("stance_proxy", {}) or {}
            stance_enabled = bool(stance_cfg.get("enabled", True))
            stance_pos = stance_cfg.get("positive") or STANCE_LEXICON["pos"]
            stance_neg = stance_cfg.get("negative") or STANCE_LEXICON["neg"]
            agent_ids = set(cfg.get("agent_ids") or [])
            user_stance_map = {}
            if stance_enabled and T_comments is not None and not T_comments.empty and not pd.isna(anchor_time):
                tmp_c = T_comments[["created_at", "comment_user", "content"]].copy()
                tmp_c["uid"] = tmp_c["comment_user"].apply(
                    lambda x: x.get("_id") if isinstance(x, dict) else None
                )
                tmp_c = tmp_c.dropna(subset=["uid"])
                if not tmp_c.empty:
                    # drop agent-authored comments
                    if agent_ids:
                        tmp_c = tmp_c[~tmp_c["uid"].astype(str).isin(agent_ids)]
                    # pre-anchor only (avoid post-treatment leakage)
                    tmp_c = tmp_c[tmp_c["created_at"] < anchor_time]
                    if not tmp_c.empty:
                        tmp_c["sent"] = tmp_c["content"].apply(
                            lambda t: _compute_sentiment(t, pos_words=stance_pos, neg_words=stance_neg)
                        )
                        # Average sentiment per user (range [-1,1])
                        user_stance_map = tmp_c.groupby("uid")["sent"].mean().to_dict()

            # Metrics (stance uses the pre-anchor stance map above)
            all_m = metrics_for(H_edges, op_id, user_stance_map)
            # Budget metrics (legacy / robustness)
            pre_m = metrics_for(pre_budget_inc_edges, op_id, user_stance_map)
            post_m = metrics_for(post_budget_inc_edges, op_id, user_stance_map)
            pre_budget_m = metrics_for(pre_budget_edges, op_id, user_stance_map)
            post_budget_m = metrics_for(post_budget_edges, op_id, user_stance_map)
            pre_time_m = metrics_for(pre_time_edges, op_id, user_stance_map)
            post_time_m = metrics_for(post_time_edges, op_id, user_stance_map)
            pre_time_inc_m = metrics_for(pre_time_inc_edges, op_id, user_stance_map)
            post_time_inc_m = metrics_for(post_time_inc_edges, op_id, user_stance_map)

            if T_comments is None or T_comments.empty or pd.isna(anchor_time):
                pre_c = T_comments.iloc[0:0] if T_comments is not None else None
                post_c = T_comments.iloc[0:0] if T_comments is not None else None
            else:
                window = pd.Timedelta(minutes=time_win)
                pre_c = T_comments[(T_comments["created_at"] < anchor_time) & (T_comments["created_at"] >= anchor_time - window)]
                post_c = T_comments[(T_comments["created_at"] >= anchor_time) & (T_comments["created_at"] <= anchor_time + window)]

            P_all = prosocial.aggregate(T_comments, "content")
            P_pre = prosocial.aggregate(pre_c, "content")
            P_post = prosocial.aggregate(post_c, "content")

            # Coverage indicators
            n_pre_main = int(len(pre_budget_inc_edges))
            n_post_main = int(len(post_budget_inc_edges))
            n_pre_budget = int(len(pre_budget_edges))
            n_post_budget = int(len(post_budget_edges))
            budget_ok = int((n_pre_budget >= min_pre) and (n_post_budget >= min_post))
            main_ok = int(
                budget_ok
                and (n_pre_main >= min_inc_pre)
                and (n_post_main >= min_inc_post)
            )

            # Time-window coverage (NEW primary filters for Sample A/B)
            n_pre_time = int(len(pre_time_edges))
            n_post_time = int(len(post_time_edges))
            n_pre_time_inc = int(len(pre_time_inc_edges))
            n_post_time_inc = int(len(post_time_inc_edges))
            mature_ok_time = int((n_pre_time >= min_pre_h) and (n_post_time >= min_post_h))

            # Thread lifecycle stage
            created_at = created_at_map.get(mid, pd.NaT)
            age_at_anchor_min = np.nan
            if not pd.isna(created_at) and not pd.isna(anchor_time):
                try:
                    age_at_anchor_min = float((anchor_time - created_at).total_seconds() / 60.0)
                except Exception:
                    age_at_anchor_min = np.nan

            # Sample label (used downstream for A/B analyses)
            if n_pre_time <= early_max_pre_h:
                sample_group = "early"
            elif mature_ok_time == 1:
                sample_group = "mature"
            else:
                sample_group = "middle"

            row = {
                "mblogid": mid,
                "tstar": tstar,
                "anchor_time": anchor_time,
                "anchor_source": anchor_source,
                "anchor_n_matches": anchor_n,
                "age_at_anchor_min": age_at_anchor_min,
                "n_edges": int(len(H_edges)),
                "n_edges_total": int(len(T_edges)),
                "n_incumbents_pre": int(len(incumbents)),
                "n_pre_budget_edges": n_pre_budget,
                "n_post_budget_edges": n_post_budget,
                "budget_ok": budget_ok,
                "n_pre_main_edges": n_pre_main,
                "n_post_main_edges": n_post_main,
                "main_budget_ok": main_ok,
                # NEW: time-window coverage + sample split
                "n_pre_time_edges": n_pre_time,
                "n_post_time_edges": n_post_time,
                "n_pre_time_inc_edges": n_pre_time_inc,
                "n_post_time_inc_edges": n_post_time_inc,
                "mature_time_ok": mature_ok_time,
                "sample_group": sample_group,
                **{f"all_{k}": v for k, v in all_m.items()},
                # Main outputs (network): incumbent-only + edge budget
                **{f"pre_{k}": v for k, v in pre_m.items()},
                **{f"post_{k}": v for k, v in post_m.items()},
                # Robustness: human-only + edge budget
                **{f"pre_budget_{k}": v for k, v in pre_budget_m.items()},
                **{f"post_budget_{k}": v for k, v in post_budget_m.items()},
                # Robustness: human-only + time window
                **{f"pre_time_{k}": v for k, v in pre_time_m.items()},
                **{f"post_time_{k}": v for k, v in post_time_m.items()},
                # NEW: incumbent-only + time window (rewiring estimand; controls entry/composition)
                **{f"pre_time_inc_{k}": v for k, v in pre_time_inc_m.items()},
                **{f"post_time_inc_{k}": v for k, v in post_time_inc_m.items()},
                # Prosocial (legacy: time window around anchor)
                **{f"all_{k}": v for k, v in P_all.items()},
                **{f"pre_{k}": v for k, v in P_pre.items()},
                **{f"post_{k}": v for k, v in P_post.items()},
            }
            rows.append(row)
            processed += 1

        out = pd.DataFrame(rows)

    # Add post-pre deltas for any metric with both columns
    if out is not None and not out.empty:
        # Save anchors separately for reuse/debugging
        try:
            ensure_dir("artifacts/matching")
            save_parquet(
                out[["mblogid", "tstar", "anchor_time", "anchor_source", "anchor_n_matches"]],
                "artifacts/matching/anchors.parquet",
            )
        except Exception:
            pass

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
