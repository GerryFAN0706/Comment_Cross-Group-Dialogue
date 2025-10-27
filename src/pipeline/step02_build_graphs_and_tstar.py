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
    posts = pd.read_parquet("artifacts/ingested/posts.parquet")
    comments = pd.read_parquet("artifacts/ingested/comments.parquet")
    threads = pd.read_parquet("artifacts/ingested/threads.parquet")

    # Edges per thread: u -> v at comment time
    all_edges = []
    tstars = []  # first agent reply time per thread
    meta = []    # per-thread pre features
    for r in tqdm(threads.itertuples(index=False), total=len(threads)):
        mblogid = r.mblogid
        op_id = r.op_id
        T = comments[comments["root_post_mblogid"] == mblogid].sort_values("created_at").copy()
        if T.empty:
            continue
        # compute edges
        T["u"] = T["comment_user"].apply(lambda u: u.get("_id") if isinstance(u, dict) else None)
        T["v"] = T["reply_comment"].apply(lambda x: x.get("user", {}).get("_id") if isinstance(x, dict) else op_id)
        T["is_agent"] = T["comment_user"].apply(lambda u: _is_agent(u, cfg))
        T["is_root_reply"] = T["reply_comment"].isna()
        # t*
        tstar = pd.NaT
        agent_rows = T[T["is_agent"]]
        if not agent_rows.empty:
            tstar = agent_rows["created_at"].iloc[0]
        tstars.append({"mblogid": mblogid, "tstar": tstar})
        # early pre-reply engagement: count non-agent comments before t*
        if pd.isna(tstar):
            early = len(T)  # if no agent, we will define pseudo later
        else:
            early_mask = (T["created_at"] < tstar) & (~T["is_agent"])
            early = early_mask.sum()
        # post-level features (pre)
        post_len = char_len(r.content or "")
        has_q = has_question(r.content or "")
        has_ht = has_hashtag(r.content or "")
        pic_num = getattr(r, "pic_num", None) or 0
        meta.append({
            "mblogid": mblogid,
            "op_id": op_id,
            "post_len": post_len,
            "post_has_question": int(has_q),
            "post_has_hashtag": int(has_ht),
            "pic_num": pic_num,
            "source": r.source,
            "created_at": r.created_at,
            "early_pre_engagement": early,
            "n_comments": len(T)
        })
        # edges table
        all_edges.append(T[["root_post_mblogid","created_at","u","v","is_agent","is_root_reply"]])

    edges = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame(columns=["root_post_mblogid","created_at","u","v","is_agent","is_root_reply"])
    tstars = pd.DataFrame(tstars)
    meta = pd.DataFrame(meta)

    save_parquet(edges, f"{ART}/edges.parquet")
    save_parquet(tstars, f"{ART}/tstars.parquet")
    save_parquet(meta, f"{ART}/thread_meta.parquet")
    print("Built edges, t* and thread_meta.")

if __name__ == "__main__":
    run()
