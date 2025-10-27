import yaml
import pandas as pd
import pyarrow as pa
from ..utils.io_utils import (
    ensure_dir,
    read_json_like,
    read_json_like_in_chunks,
    save_parquet,
    write_parquet_batches,
)
from ..utils.time_utils import parse_ts

ART = "artifacts/ingested"
COMMENTS_CHUNK_SIZE = 100_000
IP_PREFIX = "IP属地："
COMMENT_COLUMNS = [
    "created_at",
    "_id",
    "ip_location",
    "content",
    "comment_user",
    "reply_comment",
    "root_comment_id",
    "root_post_mblogid",
    "likes_count",
]

COMMENT_USER_TEMPLATE = {
    "_id": None,
    "avatar_hd": None,
    "description": None,
    "followers_count": None,
    "friends_count": None,
    "gender": None,
    "location": None,
    "mbrank": None,
    "mbtype": None,
    "nick_name": None,
    "statuses_count": None,
    "verified": None,
    "verified_reason": None,
    "verified_type": None,
    "credit_score": None,
    "sunshine_credit": None,
    "ip_location": None,
}

REPLY_USER_TEMPLATE = {
    "_id": None,
    "avatar_hd": None,
    "created_at": None,
    "credit_score": None,
    "description": None,
    "followers_count": None,
    "friends_count": None,
    "gender": None,
    "location": None,
    "mbrank": None,
    "mbtype": None,
    "nick_name": None,
    "statuses_count": None,
    "verified": None,
    "verified_reason": None,
    "verified_type": None,
    "ip_location": None,
}

REPLY_COMMENT_TEMPLATE = {
    "_id": None,
    "text": None,
    "user": REPLY_USER_TEMPLATE,
}

COMMENT_USER_TYPE = pa.struct([
    pa.field("_id", pa.string()),
    pa.field("avatar_hd", pa.string()),
    pa.field("description", pa.string()),
    pa.field("followers_count", pa.int64()),
    pa.field("friends_count", pa.int64()),
    pa.field("gender", pa.string()),
    pa.field("location", pa.string()),
    pa.field("mbrank", pa.int64()),
    pa.field("mbtype", pa.int64()),
    pa.field("nick_name", pa.string()),
    pa.field("statuses_count", pa.int64()),
    pa.field("verified", pa.bool_()),
    pa.field("verified_reason", pa.string()),
    pa.field("verified_type", pa.int64()),
    pa.field("credit_score", pa.int64()),
    pa.field("sunshine_credit", pa.string()),
    pa.field("ip_location", pa.string()),
])

REPLY_USER_TYPE = pa.struct([
    pa.field("_id", pa.string()),
    pa.field("avatar_hd", pa.string()),
    pa.field("created_at", pa.string()),
    pa.field("credit_score", pa.int64()),
    pa.field("description", pa.string()),
    pa.field("followers_count", pa.int64()),
    pa.field("friends_count", pa.int64()),
    pa.field("gender", pa.string()),
    pa.field("location", pa.string()),
    pa.field("mbrank", pa.int64()),
    pa.field("mbtype", pa.int64()),
    pa.field("nick_name", pa.string()),
    pa.field("statuses_count", pa.int64()),
    pa.field("verified", pa.bool_()),
    pa.field("verified_reason", pa.string()),
    pa.field("verified_type", pa.int64()),
    pa.field("ip_location", pa.string()),
])

REPLY_COMMENT_TYPE = pa.struct([
    pa.field("_id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("user", REPLY_USER_TYPE),
])

COMMENT_SCHEMA = pa.schema([
    pa.field("created_at", pa.timestamp("ns", tz="Asia/Shanghai")),
    pa.field("_id", pa.string()),
    pa.field("ip_location", pa.string()),
    pa.field("content", pa.string()),
    pa.field("comment_user", COMMENT_USER_TYPE),
    pa.field("reply_comment", REPLY_COMMENT_TYPE),
    pa.field("root_comment_id", pa.string()),
    pa.field("root_post_mblogid", pa.string()),
    pa.field("likes_count", pa.int64()),
])


def _coarsen_ip(df: pd.DataFrame, column: str) -> None:
    if column in df.columns:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.replace(IP_PREFIX, "", regex=False)
            .str.strip()
        )


def _normalize_dict(value, template):
    base = {k: template[k] for k in template}
    if isinstance(value, dict):
        for key, val in value.items():
            if key in base:
                base[key] = val
    return {k: base[k] for k in base}


def _harmonize_object_columns(df: pd.DataFrame) -> None:
    for col in df.select_dtypes(include="object").columns:
        series = df[col]
        canonical = None
        for value in series:
            if isinstance(value, dict):
                canonical = dict
                break
            if isinstance(value, list):
                canonical = list
                break
        if canonical is dict:
            df[col] = series.apply(lambda v: v if isinstance(v, dict) else None)
        elif canonical is list:
            df[col] = series.apply(lambda v: v if isinstance(v, list) else None)


def _process_comment_chunk(chunk: pd.DataFrame, timezone: str) -> pd.DataFrame:
    if "created_at" in chunk.columns:
        chunk["created_at"] = parse_ts(chunk["created_at"], timezone)
    _coarsen_ip(chunk, "ip_location")
    _harmonize_object_columns(chunk)
    for col in COMMENT_COLUMNS:
        if col not in chunk.columns:
            chunk[col] = pd.NA
    chunk["comment_user"] = chunk["comment_user"].apply(lambda x: _normalize_dict(x, COMMENT_USER_TEMPLATE))

    def _normalize_reply(value):
        if not isinstance(value, dict):
            value = {}
        template = {
            "_id": REPLY_COMMENT_TEMPLATE["_id"],
            "text": REPLY_COMMENT_TEMPLATE["text"],
            "user": _normalize_dict(value.get("user"), REPLY_USER_TEMPLATE)
            if isinstance(value.get("user"), dict)
            else _normalize_dict({}, REPLY_USER_TEMPLATE),
        }
        template["_id"] = value.get("_id", template["_id"])
        template["text"] = value.get("text", template["text"])
        return template

    chunk["reply_comment"] = chunk["reply_comment"].apply(_normalize_reply)

    ordered = COMMENT_COLUMNS + [c for c in chunk.columns if c not in COMMENT_COLUMNS]
    chunk = chunk[ordered]
    return chunk


def run():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    ensure_dir(ART)

    users = read_json_like("data/users.json")
    posts = read_json_like("data/posts.json")

    posts["created_at"] = parse_ts(posts["created_at"], cfg["timezone"])
    for df, col in [(users, "ip_location"), (posts, "ip_location")]:
        _coarsen_ip(df, col)
    _harmonize_object_columns(users)
    _harmonize_object_columns(posts)

    if "followers_count" in users.columns:
        users["followers_count"] = (
            pd.to_numeric(users["followers_count"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        users["follower_decile"] = pd.qcut(
            users["followers_count"], q=10, labels=False, duplicates="drop"
        )

    save_parquet(users, f"{ART}/users.parquet")
    save_parquet(posts, f"{ART}/posts.parquet")

    comment_batches = (
        _process_comment_chunk(chunk, cfg["timezone"])
        for chunk in read_json_like_in_chunks("data/comments.json", COMMENTS_CHUNK_SIZE)
    )
    write_parquet_batches(comment_batches, f"{ART}/comments.parquet", schema=COMMENT_SCHEMA)

    threads = posts[
        [
            "_id",
            "mblogid",
            "created_at",
            "user",
            "content",
            "source",
            "pic_num",
            "pic_urls",
            "likes_count",
            "comments_count",
            "reposts_count",
            "ip_location",
            "is_retweet",
            "isLongText",
        ]
    ].copy()
    threads.rename(columns={"_id": "post_id"}, inplace=True)
    threads["op_id"] = posts["user"].apply(
        lambda u: u.get("_id") if isinstance(u, dict) else None
    )
    save_parquet(threads, f"{ART}/threads.parquet")

    print("Ingest & normalize complete. Files saved under artifacts/ingested")


if __name__ == "__main__":
    run()
