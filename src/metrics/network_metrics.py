import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
from typing import Dict, Tuple

from .gini import gini

def build_digraph(edges_df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    # edges_df: columns ['u','v','time'] (users._id)
    for r in edges_df.itertuples(index=False):
        G.add_edge(r.u, r.v)
    return G

def reciprocity_stats(edges_df: pd.DataFrame, op_id=None, exclude_op=True) -> Dict[str, float]:
    """
    Reciprocity: share of dyads with two-way replies.
    If exclude_op=True, drop edges involving OP from reciprocity computation.
    """
    df = edges_df[['u','v']].copy()
    if exclude_op and op_id is not None:
        df = df[(df.u != op_id) & (df.v != op_id)]
    if len(df)==0:
        return {"R": np.nan, "R_weighted": np.nan}
    # binary dyads
    edges = set(map(tuple, df[['u','v']].itertuples(index=False, name=None)))
    mutual = 0
    # weighted: count dyads with >=2 turns each direction (approx via min counts)
    counts = Counter(map(tuple, df[['u','v']].itertuples(index=False, name=None)))
    dyads = set()
    weighted_sum = 0
    for (u,v), c_uv in counts.items():
        if (v,u) in counts:
            mutual += 1
            weighted_sum += min(c_uv, counts[(v,u)])
            dyads.add(tuple(sorted((u,v))))
    E = len(edges)
    return {
        "R": mutual / E if E>0 else np.nan,
        "R_weighted": weighted_sum / E if E>0 else np.nan
    }

def branching_factor(edges_df: pd.DataFrame, op_id) -> Dict[str, float]:
    """
    BF = (# leaves)/(# internal nodes) in the reply tree, where parent is reply target.
    We consider a rooted tree at OP; child count from 'parent -> replier' orientation.
    """
    # Flip orientation: parent is target v, child is source u
    parent_counts = edges_df.groupby('v').size().rename('children').reset_index()
    degree = pd.merge(parent_counts, parent_counts.rename(columns={'v':'node'}), how='outer',
                      left_on='v', right_on='node')
    # build node child counts
    child_counts = edges_df.groupby('v').size()
    # nodes appearing as u but not as v have 0 children
    nodes = set(edges_df.u.unique()).union(set(edges_df.v.unique()))
    children = {n: int(child_counts.get(n, 0)) for n in nodes}
    # internal nodes: child_count >=1; leaves: child_count ==0 (excluding OP from leaves by convention)
    internal = sum(1 for n,c in children.items() if c>=1)
    leaves = sum(1 for n,c in children.items() if c==0 and n!=op_id)
    bf = leaves / internal if internal>0 else float('nan')
    # simpler proxy: unique participants / #root-level replies
    root_replies = (edges_df.v==op_id).sum()
    unique_participants = len(nodes)
    proxy = unique_participants / root_replies if root_replies>0 else float('nan')
    return {"BF": bf, "BF_proxy": proxy, "leaves": leaves, "internal_nodes": internal}

def equality_of_voice(edges_df: pd.DataFrame) -> Dict[str, float]:
    counts = edges_df.groupby('u').size().values
    return {"Gini": gini(counts)}

def assortativity(edges_df: pd.DataFrame, user_attr: pd.DataFrame, attr_col: str) -> float:
    """
    Newman assortativity on categorical attributes.
    user_attr: DataFrame with ['user_id', attr_col] covering nodes in this thread.
    """
    G = nx.DiGraph()
    for r in edges_df[['u','v']].itertuples(index=False):
        G.add_edge(r.u, r.v)
    # map attribute
    attr_map = dict(zip(user_attr.user_id, user_attr[attr_col]))
    nx.set_node_attributes(G, attr_map, 'attr')
    try:
        a = nx.attribute_assortativity_coefficient(G.to_undirected(), 'attr')
    except Exception:
        a = float('nan')
    return a

def dc_bi_analytic(edges_df: pd.DataFrame, user_attr: pd.DataFrame, attr_col: str) -> Tuple[float, pd.DataFrame]:
    """
    Degree-Corrected Bridging Index (analytic expectation under configuration model).
    Groups G; E_g->h / E[E_g->h] averaged over g!=h.
    Returns (dc_bi, table of group-pair stats).
    """
    df = edges_df[['u','v']].copy()
    df = df.merge(user_attr.rename(columns={'user_id':'u'}), on='u', how='left')
    df = df.merge(user_attr.rename(columns={'user_id':'v', attr_col: attr_col+'_v'}), on='v', how='left')
    df[attr_col] = df[attr_col].fillna("unk")
    df[attr_col+'_v'] = df[attr_col+'_v'].fillna("unk")
    E = len(df)
    if E == 0:
        return float('nan'), pd.DataFrame()
    # compute degree totals by group
    out_deg = df.groupby('u').size().rename('out')
    in_deg = df.groupby('v').size().rename('in')
    ua = user_attr.set_index('user_id')[attr_col].fillna('unk')
    out_by_g = out_deg.groupby(ua).sum()
    in_by_g = in_deg.groupby(ua).sum()
    groups = sorted(set(out_by_g.index).union(set(in_by_g.index)))
    # observed E_g->h
    obs = df.groupby([attr_col, attr_col+'_v']).size().rename('E_obs').reset_index()
    rows = []
    for g in groups:
        for h in groups:
            if g == h:
                continue
            E_obs = int(obs[(obs[attr_col]==g)&(obs[attr_col+'_v']==h)]['E_obs'].sum())
            E_exp = float(out_by_g.get(g,0) * in_by_g.get(h,0)) / E if E>0 else float('nan')
            ratio = (E_obs / E_exp) if (E_exp and E_exp > 0) else float('nan')
            rows.append({"g": g, "h": h, "E_obs": E_obs, "E_exp": E_exp, "ratio": ratio})
    tab = pd.DataFrame(rows, columns=["g", "h", "E_obs", "E_exp", "ratio"])
    dc_bi = tab['ratio'].mean() if not tab.empty else float('nan')
    return dc_bi, tab
