import json
import os
from typing import Iterator, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _maybe_jsonl(path: str) -> bool:
    return path.endswith(".jsonl") or path.endswith(".ndjson")


def _iter_json_array(path: str) -> Iterator[dict]:
    """
    Stream JSON array records without loading the entire array into memory.
    """
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buffer = ""
        while True:
            chunk = f.read(1 << 20)  # 1 MiB
            if not chunk:
                eof = True
            else:
                eof = False
                buffer += chunk
            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == "[":
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "]":
                    return
                try:
                    obj, idx = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    # Need more data
                    break
                yield obj
                buffer = buffer[idx:]
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
                elif buffer.startswith("]"):
                    return
            if eof:
                if buffer.strip():
                    obj, idx = decoder.raw_decode(buffer)
                    yield obj
                break


def iter_json_like(path: str) -> Iterator[dict]:
    if _maybe_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        yield from _iter_json_array(path)


def read_json_like(path: str, limit: Optional[int] = None) -> pd.DataFrame:
    rows = []
    for idx, row in enumerate(iter_json_like(path)):
        rows.append(row)
        if limit is not None and idx + 1 >= limit:
            break
    return pd.DataFrame(rows)


def read_json_like_in_chunks(path: str, chunk_size: int) -> Iterator[pd.DataFrame]:
    rows: List[dict] = []
    for row in iter_json_like(path):
        rows.append(row)
        if len(rows) >= chunk_size:
            yield pd.DataFrame(rows)
            rows = []
    if rows:
        yield pd.DataFrame(rows)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_parquet(path, index=False)


def save_csv(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_parquet_batches(frames: Iterator[pd.DataFrame], path: str, schema: Optional[pa.Schema] = None):
    ensure_dir(os.path.dirname(path))
    writer: Optional[pq.ParquetWriter] = None
    try:
        for frame in frames:
            if frame.empty:
                continue
            current = frame.copy()
            if schema is not None:
                for field in schema.names:
                    if field not in current.columns:
                        current[field] = pd.NA
                current = current[list(schema.names)]
                table = pa.Table.from_pandas(current, schema=schema, preserve_index=False, safe=False)
            else:
                table = pa.Table.from_pandas(current, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, schema or table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
