import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import networkx as nx

from ..utils.io_utils import ensure_dir


def _load_edges(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _relabel_edges(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["u", "v"]].copy()
    df = df.dropna(subset=["u", "v"])
    return df


def _swap_edges(G: nx.Graph, num_swaps: int, max_tries: int = 1000) -> None:
    nx.double_edge_swap(G, nswap=num_swaps, max_tries=max_tries)


def _compute_dcbi(G: nx.DiGraph, attr_map: dict[str, str], groups: list[str]) -> float:
    if len(G.edges()) == 0:
        return np.nan
    total_obs = 0.0
    total_exp = 0.0
    in_deg = G.in_degree()
    out_deg = G.out_degree()
    m = G.number_of_edges()
    if m == 0:
        return np.nan
    for g in groups:
        for h in groups:
            if g == h:
                continue
            observed = 0
            expected = 0.0
            for (u, v) in G.edges():
                if attr_map.get(u) == g and attr_map.get(v) == h:
                    observed += 1
            sum_out = sum(out_deg[x] for x in G.nodes() if attr_map.get(x) == g)
            sum_in = sum(in_deg[x] for x in G.nodes() if attr_map.get(x) == h)
            expected = (sum_out * sum_in) / m if m > 0 else 0.0
            if expected > 0:
                total_obs += observed
                total_exp += expected
    if total_exp == 0:
        return np.nan
    return total_obs / total_exp


def main():
    parser = argparse.ArgumentParser(description="Compute DC-BI rewiring null statistics per thread.")
    parser.add_argument("--edges", type=Path, default=Path("artifacts/threads/edges.parquet"))
    parser.add_argument("--users", type=Path, default=Path("artifacts/ingested/users.parquet"))
    parser.add_argument("--tstars", type=Path, default=Path("artifacts/threads/tstars.parquet"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/bridging/dcbi_null.parquet"))
    parser.add_argument("--attribute", choices=["gender", "province"], default="gender")
    parser.add_argument("--nswap", type=int, default=100)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    edges = _load_edges(args.edges)
    users = pd.read_parquet(args.users)[["_id", args.attribute]].rename(columns={"_id": "user_id"})
    edges = edges.dropna(subset=["root_post_mblogid"])

    ensure_dir(args.out.parent.as_posix())
    rng = np.random.default_rng(args.seed)
    rows = []

    for thread_id, df_edges in tqdm(edges.groupby("root_post_mblogid"), desc="DC-BI rewiring"):
        df_simple = _relabel_edges(df_edges)
        if df_simple.empty:
            rows.append({"mblogid": thread_id, "attribute": args.attribute,
                         "mean": np.nan, "std": np.nan, "samples": 0})
            continue

        attr_map = pd.concat([df_simple["u"], df_simple["v"]]).unique()
        attr_map = {uid: users.loc[users["user_id"] == uid, args.attribute].fillna("unk").iloc[0]
                    for uid in attr_map if uid in set(users["user_id"])}
        groups = sorted({val for val in attr_map.values() if val != "unk"})
        if len(groups) < 2:
            rows.append({"mblogid": thread_id, "attribute": args.attribute,
                         "mean": np.nan, "std": np.nan, "samples": 0})
            continue

        base = nx.DiGraph()
        base.add_edges_from(df_simple[["u", "v"]].itertuples(index=False, name=None))

        values = []
        for _ in range(args.samples):
            H = base.copy().to_undirected()
            _swap_edges(H, args.nswap, max_tries=args.nswap * 5)
            rewired = nx.DiGraph()
            rewired.add_edges_from(H.edges())
            val = _compute_dcbi(rewired, attr_map, groups)
            if not np.isnan(val):
                values.append(val)

        if values:
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            rows.append({"mblogid": thread_id, "attribute": args.attribute,
                         "mean": mean_val, "std": std_val, "samples": len(values)})
        else:
            rows.append({"mblogid": thread_id, "attribute": args.attribute,
                         "mean": np.nan, "std": np.nan, "samples": 0})

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(args.out, index=False)
    print(f"Saved rewiring null stats to {args.out} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
