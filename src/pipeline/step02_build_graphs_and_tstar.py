import os, yaml, pandas as pd
from ..utils.io_utils import save_parquet
from ..utils.time_utils import to_minute
from ..utils.text_utils import has_question, has_hashtag, char_len
from tqdm import tqdm

ART = "artifacts/threads"

def _is_agent(user_obj, cfg):
    if not isinstance(user_obj, dict):
        return False
    uid = user_obj.get("_id")
    nick = user_obj.get("nick_name", "") or ""
    verified = bool(user_obj.get("verified", False))
    if cfg["agent_ids"] and uid in cfg["agent_ids"]:
        return True
    if cfg.get("agent_verified_required", True) and not verified:
        return False
    # nickname variants
    for v in cfg.get("agent_nickname_variants", []):
        if v in nick:
            return True
    return False

def run():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    comments = pd.read_parquet("artifacts/ingested/comments.parquet")
    threads = pd.read_parquet("artifacts/ingested/threads.parquet")

    # Build thread-level lookup (avoid repeated filtering)
    thread_lookup = threads.set_index("mblogid", drop=False)
    known_agent_ids = set(cfg.get("agent_ids") or [])

    # Edges per thread: u -> v at comment time (one row per comment)
    all_edges = []
    tstars = []  # first agent reply time per thread
    meta = []    # per-thread pre features

    if comments.empty:
        edges = pd.DataFrame(
            columns=[
                "root_post_mblogid",
                "created_at",
                "u",
                "v",
                "is_agent",
                "is_root_reply",
                "u_is_agent",
                "v_is_agent",
                "is_human_edge",
                "edge_idx",
            ]
        )
        tstars = pd.DataFrame(columns=["mblogid", "tstar"])
        meta = pd.DataFrame()
    else:
        # Sort once; groupby is much faster than filtering comments per thread in a loop
        comments_sorted = comments.sort_values(["root_post_mblogid", "created_at"])

        max_threads = int(cfg.get("debug_max_threads", 0) or 0)
        processed = 0
        for mblogid, T in tqdm(
            comments_sorted.groupby("root_post_mblogid", sort=False),
            total=comments_sorted["root_post_mblogid"].nunique(),
        ):
            if max_threads and processed >= max_threads:
                break
            if T.empty:
                continue

            # Thread-level info
            if mblogid in thread_lookup.index:
                r = thread_lookup.loc[mblogid]
                op_id = r.op_id
                post_created_at = r.created_at
                post_content = r.content or ""
                post_source = r.source
                pic_num = getattr(r, "pic_num", None) or 0
            else:
                op_id = None
                post_created_at = pd.NaT
                post_content = ""
                post_source = None
                pic_num = 0

            # Build edge endpoints
            df = T.copy()
            df["u"] = df["comment_user"].apply(
                lambda u: u.get("_id") if isinstance(u, dict) else None
            )
            df["reply_user_id"] = df["reply_comment"].apply(
                lambda rc: rc.get("user", {}).get("_id")
                if isinstance(rc, dict) and isinstance(rc.get("user"), dict)
                else None
            )
            is_root = df["reply_user_id"].isna()
            if op_id is not None:
                df.loc[is_root, "reply_user_id"] = op_id
            df["v"] = df["reply_user_id"]
            df["is_root_reply"] = is_root

            # Agent detection on source (u) and target (v)
            df["u_is_agent"] = df["comment_user"].apply(lambda u: _is_agent(u, cfg))
            # Observed agent user ids (in addition to config ids)
            observed_agent_ids = set(df.loc[df["u_is_agent"], "u"].dropna().unique().tolist())
            agent_ids = known_agent_ids.union(observed_agent_ids)
            df["v_is_agent"] = df["v"].isin(agent_ids)
            df["is_human_edge"] = (~df["u_is_agent"]) & (~df["v_is_agent"])

            # Keep legacy column name for compatibility
            df["is_agent"] = df["u_is_agent"]

            # Index within thread
            df = df.sort_values("created_at")
            df["edge_idx"] = range(len(df))

            # Drop malformed endpoints (cannot form an edge)
            df = df.dropna(subset=["u", "v"])
            if df.empty:
                # still record tstar/meta for the thread if desired
                tstars.append({"mblogid": mblogid, "tstar": pd.NaT})
                meta.append(
                    {
                        "mblogid": mblogid,
                        "op_id": op_id,
                        "post_len": char_len(post_content),
                        "post_has_question": int(has_question(post_content)),
                        "post_has_hashtag": int(has_hashtag(post_content)),
                        "pic_num": pic_num,
                        "source": post_source,
                        "created_at": post_created_at,
                        "n_comments": int(len(T)),
                        "n_edges": 0,
                        "n_human_edges": 0,
                        "early_human_comments": 0,
                    }
                )
                continue

            # t* = earliest agent-authored comment time (u_is_agent==True)
            agent_rows = df[df["u_is_agent"]]
            tstar = agent_rows["created_at"].iloc[0] if not agent_rows.empty else pd.NaT
            tstars.append({"mblogid": mblogid, "tstar": tstar})

            # Early engagement proxy for matching: human comments within Δ minutes after OP post time
            early_minutes = int(cfg.get("early_engagement_minutes", 15))
            if pd.isna(post_created_at):
                early_human_comments = int((~df["u_is_agent"]).sum())
            else:
                early_end = post_created_at + pd.Timedelta(minutes=early_minutes)
                early_human_comments = int(((df["created_at"] <= early_end) & (~df["u_is_agent"])).sum())

            meta.append(
                {
                    "mblogid": mblogid,
                    "op_id": op_id,
                    "post_len": char_len(post_content),
                    "post_has_question": int(has_question(post_content)),
                    "post_has_hashtag": int(has_hashtag(post_content)),
                    "pic_num": pic_num,
                    "source": post_source,
                    "created_at": post_created_at,
                    "n_comments": int(len(T)),
                    "n_edges": int(len(df)),
                    "n_human_edges": int(df["is_human_edge"].sum()),
                    "early_human_comments": early_human_comments,
                }
            )

            all_edges.append(
                df[
                    [
                        "root_post_mblogid",
                        "created_at",
                        "u",
                        "v",
                        "is_agent",
                        "is_root_reply",
                        "u_is_agent",
                        "v_is_agent",
                        "is_human_edge",
                        "edge_idx",
                    ]
                ]
            )

            processed += 1

        edges = (
            pd.concat(all_edges, ignore_index=True)
            if all_edges
            else pd.DataFrame(
                columns=[
                    "root_post_mblogid",
                    "created_at",
                    "u",
                    "v",
                    "is_agent",
                    "is_root_reply",
                    "u_is_agent",
                    "v_is_agent",
                    "is_human_edge",
                    "edge_idx",
                ]
            )
        )
        tstars = pd.DataFrame(tstars)
        meta = pd.DataFrame(meta)

    save_parquet(edges, f"{ART}/edges.parquet")
    save_parquet(tstars, f"{ART}/tstars.parquet")
    save_parquet(meta, f"{ART}/thread_meta.parquet")
    print("Built edges, t* and thread_meta.")

if __name__ == "__main__":
    run()
