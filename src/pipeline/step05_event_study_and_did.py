from __future__ import annotations

import math

import yaml
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..metrics.network_metrics import (
    reciprocity_stats,
    branching_factor,
    equality_of_voice,
    assortativity,
    dc_bi_analytic,
)
from ..utils.io_utils import save_parquet, ensure_dir
from ..utils.anchor_utils import build_control_anchor_map

ART = "artifacts/event_study"
MAIN_ART = "artifacts/main_effects"


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _to_naive(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        try:
            s = s.dt.tz_convert("UTC")
        except TypeError:
            s = s.dt.tz_localize("UTC")
        s = s.dt.tz_localize(None)
    return s


def _ess(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return float("nan")
    return float((w.sum() ** 2) / np.sum(w**2))


def _build_matched_weights(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-thread ATT weights from matched pairs.

    - Each treated thread gets weight 1.
    - Each treated's matched controls share total weight 1 (each pair gets 1/n_controls_for_treated).
    - Controls matched to multiple treated threads accumulate weights.
    """
    if pairs is None or pairs.empty:
        return pd.DataFrame(columns=["mblogid", "w", "role", "n_pairs"])
    if not {"mblogid_t", "mblogid_c"}.issubset(pairs.columns):
        return pd.DataFrame(columns=["mblogid", "w", "role", "n_pairs"])

    p = pairs[["mblogid_t", "mblogid_c"]].dropna().copy()
    if p.empty:
        return pd.DataFrame(columns=["mblogid", "w", "role", "n_pairs"])

    n_controls = (
        p.groupby("mblogid_t")["mblogid_c"].size().rename("n_controls").reset_index()
    )
    p = p.merge(n_controls, on="mblogid_t", how="left")
    p["pair_w"] = 1.0 / p["n_controls"].replace(0, np.nan)

    treated_npairs = p.groupby("mblogid_t").size().rename("n_pairs").reset_index()
    treated_w = p[["mblogid_t"]].drop_duplicates().rename(
        columns={"mblogid_t": "mblogid"}
    )
    treated_w = treated_w.merge(
        treated_npairs.rename(columns={"mblogid_t": "mblogid"}), on="mblogid", how="left"
    )
    treated_w["w"] = 1.0
    treated_w["role"] = "treated"

    control_w = (
        p.groupby("mblogid_c")["pair_w"].sum().rename("w").reset_index().rename(
            columns={"mblogid_c": "mblogid"}
        )
    )
    control_npairs = p.groupby("mblogid_c").size().rename("n_pairs").reset_index().rename(
        columns={"mblogid_c": "mblogid"}
    )
    control_w = control_w.merge(control_npairs, on="mblogid", how="left")
    control_w["role"] = "control"

    out = pd.concat([treated_w[["mblogid", "w", "role", "n_pairs"]], control_w], ignore_index=True)
    out["w"] = pd.to_numeric(out["w"], errors="coerce").fillna(0.0)
    out["n_pairs"] = pd.to_numeric(out["n_pairs"], errors="coerce").fillna(0).astype(int)
    out = out[out["w"] > 0].copy()
    return out


def _make_mu0_model(cfg: dict, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    alpha = float(cfg.get("dr_ridge_alpha", 1.0))
    numeric_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", numeric_tf, numeric_cols),
            ("cat", cat_tf, categorical_cols),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("model", Ridge(alpha=alpha))])


def _crossfit_mu0(
    X: pd.DataFrame,
    y: pd.Series,
    treated: pd.Series,
    w: pd.Series,
    *,
    cfg: dict,
) -> np.ndarray:
    yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    D = treated.astype(int).to_numpy(dtype=int)
    ww = pd.to_numeric(w, errors="coerce").fillna(1.0).to_numpy(dtype=float)

    n_splits = int(cfg.get("dr_n_splits", 5))
    seed = int(cfg.get("seed", 2025))

    n_t = int(D.sum())
    n_c = int((1 - D).sum())
    n_splits_eff = int(min(n_splits, n_t, n_c))
    if n_splits_eff < 2:
        # Fall back to in-sample fit on controls
        ctrl = (D == 0) & np.isfinite(yv)
        if not np.any(ctrl):
            return np.full(len(X), np.nan, dtype=float)
        # simple: constant mean of controls
        mu0_const = float(np.average(yv[ctrl], weights=ww[ctrl]))
        return np.full(len(X), mu0_const, dtype=float)

    numeric_cols = [
        c
        for c in [
            "post_len",
            "post_has_question",
            "post_has_hashtag",
            "pic_num",
            "hour_of_day",
            "day_of_week",
            "early_human_comments",
            "propensity",
        ]
        if c in X.columns
    ]
    categorical_cols = [c for c in ["source"] if c in X.columns]
    if not numeric_cols and not categorical_cols:
        ctrl = (D == 0) & np.isfinite(yv)
        if not np.any(ctrl):
            return np.full(len(X), np.nan, dtype=float)
        mu0_const = float(np.average(yv[ctrl], weights=ww[ctrl]))
        return np.full(len(X), mu0_const, dtype=float)
    model_template = _make_mu0_model(cfg, numeric_cols, categorical_cols)

    mu0 = np.full(len(X), np.nan, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, D):
        train_ctrl = train_idx[D[train_idx] == 0]
        if train_ctrl.size < 5:
            continue
        model = clone(model_template)
        model.fit(
            X.iloc[train_ctrl],
            yv[train_ctrl],
            model__sample_weight=ww[train_ctrl],
        )
        mu0[test_idx] = model.predict(X.iloc[test_idx])

    # Fill any missing predictions with weighted mean of controls
    ctrl = (D == 0) & np.isfinite(yv)
    fallback = float(np.average(yv[ctrl], weights=ww[ctrl])) if np.any(ctrl) else np.nan
    mu0[~np.isfinite(mu0)] = fallback
    return mu0


def _bootstrap_att(
    score: np.ndarray,
    treated: np.ndarray,
    w: np.ndarray,
    *,
    B: int,
    seed: int,
) -> np.ndarray:
    if B <= 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    n = int(len(score))
    out = np.full(B, np.nan, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        denom = float(np.sum(w[idx] * treated[idx]))
        if denom <= 0:
            continue
        out[b] = float(np.sum(w[idx] * score[idx]) / denom)
    return out


def _dr_att_crossfit(
    df: pd.DataFrame,
    outcome_col: str,
    *,
    cfg: dict,
    covariate_cols: list[str],
    weight_col: str = "w",
    treated_col: str = "treated",
    propensity_col: str = "propensity",
) -> dict:
    cols = [outcome_col, treated_col, propensity_col, weight_col] + covariate_cols
    # de-duplicate while preserving order
    cols = list(dict.fromkeys(cols))
    d = df[cols].copy()
    d[outcome_col] = pd.to_numeric(d[outcome_col], errors="coerce")
    d[propensity_col] = pd.to_numeric(d[propensity_col], errors="coerce")
    d[weight_col] = pd.to_numeric(d[weight_col], errors="coerce").fillna(1.0)
    d = d.dropna(subset=[outcome_col, propensity_col])
    # Guard against inf/-inf, which can silently poison sums (e.g., inf - inf -> NaN)
    d = d[np.isfinite(d[outcome_col]) & np.isfinite(d[propensity_col])].copy()

    if len(d) < 30 or d[treated_col].nunique() < 2:
        n = int(len(d))
        if n:
            D0 = pd.to_numeric(d[treated_col], errors="coerce").fillna(0).astype(int).to_numpy()
            w0 = pd.to_numeric(d[weight_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
            n_t = int(D0.sum())
            n_c = int((1 - D0).sum())
            ess_all = _ess(w0)
            ess_t = _ess(w0[D0 == 1])
            ess_c = _ess(w0[D0 == 0])
        else:
            n_t = 0
            n_c = 0
            ess_all = np.nan
            ess_t = np.nan
            ess_c = np.nan
        return {
            "outcome": outcome_col,
            "estimator": "matched_dr_aipw",
            "att": np.nan,
            "se": np.nan,
            "p": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": n,
            "n_treated": n_t,
            "n_control": n_c,
            "ess": ess_all,
            "ess_treated": ess_t,
            "ess_control": ess_c,
            "propensity_clip": float(cfg.get("dr_propensity_clip", 0.01)),
            "n_splits": int(cfg.get("dr_n_splits", 5)),
            "bootstrap_B": int(cfg.get("dr_bootstrap_B", 200) or 0),
        }

    clip = float(cfg.get("dr_propensity_clip", 0.01))
    e = np.clip(d[propensity_col].to_numpy(dtype=float), clip, 1.0 - clip)

    X = d[covariate_cols].copy()
    mu0 = _crossfit_mu0(
        X,
        d[outcome_col],
        d[treated_col],
        d[weight_col],
        cfg=cfg,
    )

    y = d[outcome_col].to_numpy(dtype=float)
    D = d[treated_col].astype(int).to_numpy(dtype=int)
    w = d[weight_col].to_numpy(dtype=float)

    score = D * (y - mu0) - (1 - D) * (e / (1.0 - e)) * (y - mu0)
    ok = np.isfinite(score) & np.isfinite(w) & np.isfinite(e)
    score = score[ok]
    D = D[ok]
    w = w[ok]
    denom = float(np.sum(w * D))
    att = float(np.sum(w * score) / denom) if denom > 0 else np.nan

    B = int(cfg.get("dr_bootstrap_B", 200) or 0)
    boot_seed = int(cfg.get("dr_bootstrap_seed", cfg.get("seed", 2025)))
    boot = _bootstrap_att(score, D, w, B=B, seed=boot_seed)
    se = float(np.nanstd(boot, ddof=1)) if np.isfinite(att) and boot.size > 1 else np.nan
    if boot.size >= 20:
        ci_low = float(np.nanquantile(boot, 0.025))
        ci_high = float(np.nanquantile(boot, 0.975))
    else:
        ci_low = np.nan
        ci_high = np.nan

    if se and np.isfinite(se) and se > 0 and np.isfinite(att):
        z = float(att / se)
        p = float(2.0 * (1.0 - _norm_cdf(abs(z))))
    else:
        p = np.nan

    ess_all = _ess(w)
    ess_t = _ess(w[D == 1])
    ess_c = _ess(w[D == 0])
    return {
        "outcome": outcome_col,
        "estimator": "matched_dr_aipw",
        "att": att,
        "se": se,
        "p": p,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(len(d)),
        "n_treated": int(D.sum()),
        "n_control": int((1 - D).sum()),
        "ess": ess_all,
        "ess_treated": ess_t,
        "ess_control": ess_c,
        "propensity_clip": clip,
        "n_splits": int(cfg.get("dr_n_splits", 5)),
        "bootstrap_B": B,
    }


def _slice_edge_bin(pre_edges: pd.DataFrame, post_edges: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """
    Slice a fixed edge-index bin relative to anchor split.
    - pre bins: start<0 and end<=0, counted backwards from anchor
    - post bins: start>=0 and end>0, counted forwards from anchor
    """
    if end <= 0 and start < 0:
        k1 = int(-start)
        k2 = int(-end)
        if k1 <= k2 or k1 <= 0 or k2 < 0:
            return pre_edges.iloc[0:0]
        tail = pre_edges.tail(k1)
        return tail.head(k1 - k2)
    if start >= 0 and end > 0:
        return post_edges.iloc[int(start) : int(end)]
    return pre_edges.iloc[0:0]


def _metrics_for_bin(
    df_edges: pd.DataFrame,
    *,
    op_id: str | None,
    ua_province: pd.DataFrame,
    exclude_op_recip: bool,
) -> dict:
    if df_edges is None or df_edges.empty:
        return {
            "R": np.nan,
            "R_weighted": np.nan,
            "BF": np.nan,
            "BF_proxy": np.nan,
            "Gini": np.nan,
            "Assort_province": np.nan,
            "DCBI_province": np.nan,
        }

    base = df_edges[["u", "v"]].copy()
    out = {}
    out.update(reciprocity_stats(base, op_id=op_id, exclude_op=exclude_op_recip))
    bf = branching_factor(base, op_id=op_id)
    out["BF"] = bf.get("BF", np.nan)
    out["BF_proxy"] = bf.get("BF_proxy", np.nan)
    out.update(equality_of_voice(base))

    nodes = pd.unique(base[["u", "v"]].values.ravel("K"))
    uattr = ua_province[ua_province["user_id"].isin(nodes)].copy()
    if uattr.empty:
        out["Assort_province"] = np.nan
        out["DCBI_province"] = np.nan
    else:
        out["Assort_province"] = assortativity(base, uattr, "province")
        dcbi, _ = dc_bi_analytic(base, uattr, "province")
        out["DCBI_province"] = dcbi
    return out


def _bin_end(label: str) -> int | None:
    try:
        return int(str(label).split(":")[1])
    except Exception:
        return None


def _term_to_bin(term: str) -> str | None:
    """
    Extract bin label from a statsmodels term like:
      - treated:C(bin)[-20:0]
      - treated:C(bin)[T.-20:0]
    """
    s = str(term)
    if "treated:C(bin)" not in s:
        return None
    if "[T." in s:
        return s.split("[T.", 1)[1].rstrip("]")
    if "[" in s:
        return s.split("[", 1)[1].rstrip("]")
    return None


def run():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    _ = cfg.get("timezone", "UTC")
    seed = int(cfg.get("seed", 2025))
    edges = pd.read_parquet("artifacts/threads/edges.parquet")
    tstars = pd.read_parquet("artifacts/threads/tstars.parquet")
    try:
        # Prefer the new strict analysis population if available
        pairs = pd.read_parquet("artifacts/matching/valid_pairs.parquet")
    except Exception:
        try:
            pairs = pd.read_parquet("artifacts/matching/matched_pairs.parquet")
        except Exception:
            pairs = pd.DataFrame()
    outcomes = pd.read_parquet("artifacts/outcomes/outcomes.parquet")
    threads = pd.read_parquet("artifacts/ingested/threads.parquet")

    if edges.empty:
        ensure_dir(ART)
        save_parquet(pd.DataFrame(), f"{ART}/binned_edges.parquet")
        save_parquet(pd.DataFrame(), f"{ART}/did_results.parquet")
        print("Event study skipped: no edges available.")
        return
    print(f"Event study: processing {len(edges):,} edges across {edges['root_post_mblogid'].nunique():,} threads.")

    # -------------------------
    # (A) Legacy time-binned edge summary (minutes)
    # -------------------------
    # Prefer unified anchor_time if available (treated: t*, controls: matched-median / fallback)
    anchor_map = {}
    if outcomes is not None and not outcomes.empty and "anchor_time" in outcomes.columns:
        anchor_map = dict(zip(outcomes["mblogid"].astype(str), outcomes["anchor_time"]))
    else:
        tmap = dict(zip(tstars["mblogid"].astype(str), tstars["tstar"]))
        anchor_map = tmap
        created_at_map = {}
        if threads is not None and not threads.empty and "mblogid" in threads.columns and "created_at" in threads.columns:
            created_at_map = dict(zip(threads["mblogid"].astype(str), threads["created_at"]))
        control_anchor_map, _ = build_control_anchor_map(
            pairs,
            tmap,
            strategy=str(cfg.get("control_anchor_strategy", "matched_median")),
            created_at_map=created_at_map,
        )
        if control_anchor_map:
            anchor_map = {**anchor_map, **control_anchor_map}

    edges["tstar"] = edges["root_post_mblogid"].astype(str).map(anchor_map)

    # Keep only edges with a valid anchor (treated or pseudo for controls)
    edges = edges[edges["tstar"].notna()].copy()

    # Prefer human-only edges if available (remove agent endpoints)
    if "is_human_edge" in edges.columns:
        edges = edges[edges["is_human_edge"]].copy()

    edges["created_at_local"] = _to_naive(edges["created_at"])
    edges["tstar_local"] = _to_naive(edges["tstar"])
    edges["tau_min"] = (edges["created_at_local"] - edges["tstar_local"]).dt.total_seconds() / 60.0

    bins = cfg["event_bins_minutes"]
    labels = [f"[{bins[i]},{bins[i+1]})" for i in range(len(bins) - 1)]
    edges["bin"] = pd.cut(
        edges["tau_min"],
        bins=[-1e9] + bins + [1e9],
        labels=["<min"] + labels + [">=max"],
        right=False,
    )

    agg = (
        edges.groupby(["root_post_mblogid", "bin"], observed=True)
        .agg(n_edges=("u", "size"), root_replies=("is_root_reply", "sum"))
        .reset_index()
    )

    thread_flags = threads[["mblogid"]].copy()
    thread_flags["treated"] = threads["mblogid"].isin(tstars[tstars["tstar"].notna()]["mblogid"])
    agg = agg.merge(thread_flags, left_on="root_post_mblogid", right_on="mblogid", how="left")
    agg["root_reply_rate"] = np.where(agg["n_edges"] > 0, agg["root_replies"] / agg["n_edges"], np.nan)
    ensure_dir(ART)
    save_parquet(agg, f"{ART}/binned_edges.parquet")

    event_summary = (
        agg.groupby(["bin", "treated"], observed=True)
        .agg(
            mean_edges=("n_edges", "mean"),
            mean_root_reply_rate=("root_reply_rate", "mean"),
            n_threads=("root_post_mblogid", "nunique")
        )
        .reset_index()
    )
    save_parquet(event_summary, f"{ART}/event_summary.parquet")

    # -------------------------
    # (B) OLS robustness regressions (NEW A/B estimands)
    # -------------------------
    df = outcomes.copy()
    # Keep sample consistent with strict matched anchors:
    # restrict to treated + their matched controls (avoid any synthetic thread-median anchors).
    matched_ids = set()
    if pairs is not None and not pairs.empty and {"mblogid_t", "mblogid_c"}.issubset(pairs.columns):
        matched_ids = set(pairs["mblogid_t"].dropna().astype(str)).union(
            set(pairs["mblogid_c"].dropna().astype(str))
        )
    if matched_ids and "mblogid" in df.columns:
        df = df[df["mblogid"].astype(str).isin(matched_ids)].copy()
    if "mblogid" in df.columns:
        df["mblogid"] = df["mblogid"].astype(str)
    if "anchor_source" in df.columns:
        df = df[
            df["anchor_source"]
            .astype(str)
            .isin(["tstar", "matched_median", "matched_median_latency"])
        ].copy()
    # IMPORTANT: keep treated as numeric 0/1 so OLS coefficient name is stable ("treated")
    df["treated"] = df["tstar"].notna().astype(int)

    threads_ct = threads[["mblogid", "created_at"]].copy()
    threads_ct["mblogid"] = threads_ct["mblogid"].astype(str)
    threads_ct["created_at_local"] = _to_naive(threads_ct["created_at"])
    df = df.merge(threads_ct[["mblogid", "created_at_local"]], on="mblogid", how="left")
    df["hour"] = df["created_at_local"].dt.hour
    df["dow"] = df["created_at_local"].dt.dayofweek

    post_outcomes = cfg.get("dr_primary_post_outcomes") or []
    delta_outcomes = cfg.get("dr_primary_delta_outcomes") or []

    results = []

    # Sample A (formation): post-only levels on time window
    df_post = df.copy()
    if "sample_group" in df_post.columns:
        df_post = df_post[df_post["sample_group"].astype(str) == "early"].copy()
    if "n_post_time_edges" in df_post.columns:
        df_post = df_post[pd.to_numeric(df_post["n_post_time_edges"], errors="coerce").fillna(0) > 0].copy()
    for col in [c for c in post_outcomes if c in df_post.columns]:
        sub = df_post[[col, "treated", "hour", "dow"]].dropna().copy()
        if len(sub) < 30 or sub["treated"].nunique() < 2:
            continue
        sub["treated"] = pd.to_numeric(sub["treated"], errors="coerce").fillna(0).astype(int)
        model = smf.ols(f"{col} ~ treated + C(hour):C(dow)", data=sub).fit(cov_type="HC1")
        results.append(
            {
                "sample": "post_only",
                "outcome": col,
                "coef_treated": float(model.params.get("treated", float("nan"))),
                "se": float(model.bse.get("treated", float("nan"))),
                "p": float(model.pvalues.get("treated", float("nan"))),
                "n": int(len(sub)),
            }
        )

    # Sample B (rewiring): DID on mature threads using incumbent-only time-window deltas
    df_mature = df.copy()
    if "sample_group" in df_mature.columns:
        df_mature = df_mature[df_mature["sample_group"].astype(str) == "mature"].copy()
    elif "mature_time_ok" in df_mature.columns:
        df_mature = df_mature[pd.to_numeric(df_mature["mature_time_ok"], errors="coerce").fillna(0).astype(int) == 1].copy()
    for col in [c for c in delta_outcomes if c in df_mature.columns]:
        sub = df_mature[[col, "treated", "hour", "dow"]].dropna().copy()
        if len(sub) < 30 or sub["treated"].nunique() < 2:
            continue
        sub["treated"] = pd.to_numeric(sub["treated"], errors="coerce").fillna(0).astype(int)
        model = smf.ols(f"{col} ~ treated + C(hour):C(dow)", data=sub).fit(cov_type="HC1")
        results.append(
            {
                "sample": "mature_did",
                "outcome": col,
                "coef_treated": float(model.params.get("treated", float("nan"))),
                "se": float(model.bse.get("treated", float("nan"))),
                "p": float(model.pvalues.get("treated", float("nan"))),
                "n": int(len(sub)),
            }
        )

    did_df = pd.DataFrame(results)
    save_parquet(did_df, f"{ART}/did_results.parquet")

    # -------------------------
    # (C) Matched DR (AIPW) on dY with cross-fitting (MAIN estimator)
    # -------------------------
    weights = _build_matched_weights(pairs)
    
    # Apply weight trimming to improve ESS and DR stability
    max_w = float(cfg.get("dr_max_weight", 0))
    if max_w > 0 and not weights.empty:
        n_trimmed = int((weights["w"] > max_w).sum())
        if n_trimmed > 0:
            print(f"Weight trimming: clipping {n_trimmed:,} weights > {max_w} to {max_w}")
        weights["w"] = weights["w"].clip(upper=max_w)
    
    if not weights.empty:
        try:
            prop = pd.read_parquet("artifacts/matching/propensity.parquet")
        except Exception:
            prop = pd.DataFrame(columns=["mblogid", "propensity"])
        w_save = weights.merge(prop, on="mblogid", how="left")
        ensure_dir("artifacts/matching")
        save_parquet(w_save, "artifacts/matching/weights.parquet")

    try:
        meta = pd.read_parquet("artifacts/threads/thread_meta.parquet")
    except Exception:
        meta = pd.DataFrame()

    dr_post_rows: list[dict] = []
    dr_mature_rows: list[dict] = []
    if weights.empty or df.empty:
        dr_post_df = pd.DataFrame()
        dr_mature_df = pd.DataFrame()
    else:
        # Build covariates X from strictly pre-treatment thread_meta (+ propensity as a feature)
        cov = meta.copy()
        if "mblogid" in cov.columns:
            cov["mblogid"] = cov["mblogid"].astype(str)
        if "created_at" in cov.columns:
            cov["created_at_local"] = _to_naive(cov["created_at"])
        else:
            cov["created_at_local"] = pd.Series([pd.NaT] * len(cov))
        cov["hour_of_day"] = cov["created_at_local"].dt.hour
        cov["day_of_week"] = cov["created_at_local"].dt.dayofweek
        if "source" in cov.columns:
            cov["source"] = cov["source"].astype(str).fillna("")
        else:
            cov["source"] = ""
        for col in ["post_len", "post_has_question", "post_has_hashtag", "pic_num", "early_human_comments"]:
            if col in cov.columns:
                cov[col] = pd.to_numeric(cov[col], errors="coerce")
        cov = cov[
            [
                c
                for c in [
                    "mblogid",
                    "post_len",
                    "post_has_question",
                    "post_has_hashtag",
                    "pic_num",
                    "hour_of_day",
                    "day_of_week",
                    "early_human_comments",
                    "source",
                ]
                if c in cov.columns
            ]
        ].drop_duplicates("mblogid")

        try:
            prop = pd.read_parquet("artifacts/matching/propensity.parquet")
        except Exception:
            prop = pd.DataFrame(columns=["mblogid", "propensity"])
        if "mblogid" in prop.columns:
            prop["mblogid"] = prop["mblogid"].astype(str)

        df_dr = df.merge(weights[["mblogid", "w"]], on="mblogid", how="inner")
        df_dr = df_dr.merge(prop, on="mblogid", how="left")
        df_dr = df_dr.merge(cov, on="mblogid", how="left")
        df_dr["propensity"] = pd.to_numeric(df_dr["propensity"], errors="coerce")

        covariate_cols = [c for c in cov.columns if c != "mblogid"]
        # Allow the outcome model to use the (precomputed) propensity as a summary feature
        if "propensity" in df_dr.columns and "propensity" not in covariate_cols:
            covariate_cols = covariate_cols + ["propensity"]

        # ---- Sample A: post-only (formation) ----
        df_post_dr = df_dr.copy()
        if "sample_group" in df_post_dr.columns:
            df_post_dr = df_post_dr[df_post_dr["sample_group"].astype(str) == "early"].copy()
        if "n_post_time_edges" in df_post_dr.columns:
            df_post_dr = df_post_dr[
                pd.to_numeric(df_post_dr["n_post_time_edges"], errors="coerce").fillna(0) > 0
            ].copy()
        post_list = cfg.get("dr_primary_post_outcomes") or []
        post_list = [c for c in post_list if c in df_post_dr.columns]
        for outcome_col in post_list:
            row = _dr_att_crossfit(
                df_post_dr,
                outcome_col,
                cfg=cfg,
                covariate_cols=covariate_cols,
                weight_col="w",
                treated_col="treated",
                propensity_col="propensity",
            )
            row["sample"] = "post_only"
            dr_post_rows.append(row)
        dr_post_df = pd.DataFrame(dr_post_rows)

        # ---- Sample B: mature DID (rewiring) ----
        df_mature_dr = df_dr.copy()
        if "sample_group" in df_mature_dr.columns:
            df_mature_dr = df_mature_dr[df_mature_dr["sample_group"].astype(str) == "mature"].copy()
        elif "mature_time_ok" in df_mature_dr.columns:
            df_mature_dr = df_mature_dr[
                pd.to_numeric(df_mature_dr["mature_time_ok"], errors="coerce").fillna(0).astype(int) == 1
            ].copy()

        delta_list = cfg.get("dr_primary_delta_outcomes") or []
        delta_list = [c for c in delta_list if c in df_mature_dr.columns]
        for outcome_col in delta_list:
            row = _dr_att_crossfit(
                df_mature_dr,
                outcome_col,
                cfg=cfg,
                covariate_cols=covariate_cols,
                weight_col="w",
                treated_col="treated",
                propensity_col="propensity",
            )
            row["sample"] = "mature"
            dr_mature_rows.append(row)
        dr_mature_df = pd.DataFrame(dr_mature_rows)

    ensure_dir(MAIN_ART)
    save_parquet(dr_mature_df, f"{MAIN_ART}/att_main.parquet")
    save_parquet(dr_post_df, f"{MAIN_ART}/att_post_main.parquet")

    # -------------------------
    # (D) Edge-index event study around anchor (main spec, incumbent-only)
    # -------------------------
    edge_bins = cfg.get("event_edge_bins") or [-60, -40, -20, 0, 20, 40, 60]
    try:
        edge_bins = sorted({int(x) for x in edge_bins})
    except Exception:
        edge_bins = [-60, -40, -20, 0, 20, 40, 60]
    if 0 not in edge_bins:
        edge_bins.append(0)
        edge_bins = sorted(edge_bins)
    intervals = [(edge_bins[i], edge_bins[i + 1]) for i in range(len(edge_bins) - 1)]
    intervals = [(a, b) for (a, b) in intervals if not (a < 0 < b)]

    min_edges = int(cfg.get("event_edge_min_edges", 5))
    include_op = bool(cfg.get("incumbent_include_op", True))
    exclude_op_recip = bool(cfg.get("exclude_op_in_reciprocity_main", True))

    # User attribute table (province)
    try:
        users = pd.read_parquet("artifacts/ingested/users.parquet")
        ua = users[["_id", "ip_location"]].copy()
        ua.rename(columns={"_id": "user_id", "ip_location": "province"}, inplace=True)
        ua["province"] = ua["province"].fillna("unk").replace({"": "unk"})
    except Exception:
        ua = pd.DataFrame(columns=["user_id", "province"])

    # OP map
    op_map = {}
    if not meta.empty and "mblogid" in meta.columns and "op_id" in meta.columns:
        op_map = dict(zip(meta["mblogid"].astype(str), meta["op_id"]))
    elif "mblogid" in threads.columns and "op_id" in threads.columns:
        op_map = dict(zip(threads["mblogid"].astype(str), threads["op_id"]))

    # Matched sample restriction (recommended for consistency with main estimator)
    matched_ids = set(weights["mblogid"]) if not weights.empty else set()
    w_map = dict(zip(weights["mblogid"], weights["w"])) if not weights.empty else {}
    treated_ids = (
        set(tstars.loc[tstars["tstar"].notna(), "mblogid"].astype(str)) if not tstars.empty else set()
    )

    edges_es = edges.copy()
    if matched_ids:
        edges_es = edges_es[edges_es["root_post_mblogid"].astype(str).isin(matched_ids)].copy()
    # NEW: event study is only meaningful on the mature subsample (pre-state exists)
    mature_ids = set()
    if "sample_group" in df.columns:
        mature_ids = set(df.loc[df["sample_group"].astype(str) == "mature", "mblogid"].astype(str).tolist())
    elif "mature_time_ok" in df.columns:
        mature_ids = set(
            df.loc[pd.to_numeric(df["mature_time_ok"], errors="coerce").fillna(0).astype(int) == 1, "mblogid"]
            .astype(str)
            .tolist()
        )
    if mature_ids:
        edges_es = edges_es[edges_es["root_post_mblogid"].astype(str).isin(mature_ids)].copy()
    edges_es["root_post_mblogid"] = edges_es["root_post_mblogid"].astype(str)
    if "is_human_edge" in edges_es.columns:
        edges_es = edges_es[edges_es["is_human_edge"]].copy()
    if "created_at_local" not in edges_es.columns and "created_at" in edges_es.columns:
        edges_es["created_at_local"] = _to_naive(edges_es["created_at"])

    sort_cols = ["root_post_mblogid", "created_at_local" if "created_at_local" in edges_es.columns else "created_at"]
    if "edge_idx" in edges_es.columns:
        sort_cols.append("edge_idx")
    edges_es = edges_es.sort_values(sort_cols)

    max_threads = int(cfg.get("debug_max_threads", 0) or 0)
    processed = 0
    rows = []
    for mid, E in edges_es.groupby("root_post_mblogid", sort=False):
        if max_threads and processed >= max_threads:
            break
        anchor_time = anchor_map.get(mid, pd.NaT)
        anchor_time_local = _to_naive(pd.Series([anchor_time])).iloc[0]
        if pd.isna(anchor_time):
            # fallback: thread median edge time
            ordered = pd.to_datetime(
                E["created_at_local"] if "created_at_local" in E.columns else E["created_at"],
                errors="coerce",
            ).dropna().sort_values()
            anchor_time_local = ordered.iloc[(len(ordered) - 1) // 2] if len(ordered) else pd.NaT
        if pd.isna(anchor_time_local):
            continue

        sort_time_col = "created_at_local" if "created_at_local" in E.columns else "created_at"
        E = E.sort_values([sort_time_col] + (["edge_idx"] if "edge_idx" in E.columns else []))
        pre = E[E[sort_time_col] < anchor_time_local]
        post = E[E[sort_time_col] >= anchor_time_local]
        incumbents = set(pd.unique(pre[["u", "v"]].values.ravel("K")))
        op_id = op_map.get(mid)
        if include_op and op_id is not None:
            incumbents.add(op_id)
        if not incumbents:
            continue

        treated_flag = bool(mid in treated_ids)
        w_i = float(w_map.get(mid, 1.0))

        for a, b in intervals:
            seg = _slice_edge_bin(pre, post, a, b)
            if seg.empty:
                continue
            seg_inc = seg[seg["u"].isin(incumbents) & seg["v"].isin(incumbents)].copy()
            m = _metrics_for_bin(seg_inc, op_id=op_id, ua_province=ua, exclude_op_recip=exclude_op_recip)
            rows.append(
                {
                    "root_post_mblogid": mid,
                    "treated": treated_flag,
                    "w": w_i,
                    "bin_start": int(a),
                    "bin_end": int(b),
                    "bin": f"{int(a)}:{int(b)}",
                    "n_edges_bin": int(len(seg)),
                    "n_edges_incumbent": int(len(seg_inc)),
                    **m,
                }
            )
        processed += 1

    panel = pd.DataFrame(rows)
    save_parquet(panel, f"{ART}/event_panel_edge_main.parquet")

    coeff_rows = []
    cov_rows = []
    metrics = cfg.get("event_edge_metrics") or ["DCBI_province", "Assort_province", "R", "BF", "Gini"]
    metrics = [m for m in metrics if m in panel.columns] if not panel.empty else []
    if not panel.empty and metrics:
        for metric in metrics:
            sub = panel[
                (panel["n_edges_incumbent"] >= min_edges)
                & panel[metric].notna()
                & panel["bin"].notna()
            ][["root_post_mblogid", "treated", "bin", "w", metric]].copy()
            if len(sub) < 50 or sub["treated"].nunique() < 2:
                continue
            sub["treated"] = sub["treated"].astype(int)
            # Normalize weights so each thread contributes total weight ~= w across bins
            nbin = sub.groupby("root_post_mblogid")["bin"].transform("nunique").clip(lower=1)
            sub["w_row"] = sub["w"] / nbin
            try:
                mfit = smf.wls(
                    f"{metric} ~ 0 + C(bin) + treated:C(bin)",
                    data=sub,
                    weights=sub["w_row"],
                ).fit(cov_type="cluster", cov_kwds={"groups": sub["root_post_mblogid"]})

                # Save treated-by-bin coefficients
                for term, coef in mfit.params.items():
                    if "treated:C(bin)" not in term:
                        continue
                    se = float(mfit.bse.get(term, np.nan))
                    p = float(mfit.pvalues.get(term, np.nan))
                    # term examples:
                    #   treated:C(bin)[-20:0]
                    #   treated:C(bin)[T.-20:0]
                    if "[T." in term:
                        bin_label = term.split("[T.", 1)[1].rstrip("]")
                    else:
                        bin_label = term.split("[", 1)[1].rstrip("]")
                    coeff_rows.append(
                        {
                            "metric": metric,
                            "bin": bin_label,
                            "coef": float(coef),
                            "se": se,
                            "p": p,
                            "n_rows": int(len(sub)),
                            "n_threads": int(sub["root_post_mblogid"].nunique()),
                        }
                    )

                cov = mfit.cov_params()
                treated_terms = [c for c in cov.index if "treated:C(bin)" in str(c)]
                if treated_terms:
                    cov_sub = cov.loc[treated_terms, treated_terms]
                    tmp = (
                        cov_sub.stack()
                        .reset_index()
                        .rename(columns={"level_0": "term1", "level_1": "term2", 0: "cov"})
                    )
                    tmp["metric"] = metric
                    cov_rows.append(tmp)
            except Exception:
                continue

    event_coeffs = pd.DataFrame(coeff_rows)
    if not event_coeffs.empty:
        event_coeffs["sample"] = "mature"
    save_parquet(event_coeffs, f"{ART}/event_coeffs_edge_main.parquet")
    if cov_rows:
        event_cov = pd.concat(cov_rows, ignore_index=True)
    else:
        event_cov = pd.DataFrame()
    if not event_cov.empty:
        event_cov["sample"] = "mature"
    save_parquet(event_cov, f"{ART}/event_cov_edge_main.parquet")

    # Also provide baseline-relative (standard event-study style) coefficients for easier plotting
    # coef_rel(bin) = coef(bin) - coef(baseline_pre_bin)
    rel_rows = []
    rel_cov_rows = []
    if not event_coeffs.empty and not event_cov.empty:
        for metric, g in event_coeffs.groupby("metric", sort=False):
            gg = g.copy()
            gg["bin_end"] = gg["bin"].map(_bin_end)
            pre = gg[gg["bin_end"].notna() & (gg["bin_end"] <= 0)]
            if pre.empty:
                continue
            # baseline: closest pre bin to 0 (e.g., -20:0)
            baseline_bin = str(pre.sort_values("bin_end", ascending=False).iloc[0]["bin"])
            if baseline_bin not in set(gg["bin"].astype(str)):
                continue
            base_coef = float(gg.loc[gg["bin"].astype(str) == baseline_bin, "coef"].iloc[0])

            # Build term map from covariance table
            covg = event_cov[event_cov["metric"] == metric].copy()
            if covg.empty:
                continue
            terms = sorted(set(covg["term1"]).union(set(covg["term2"])))
            term_by_bin = {}
            for t in terms:
                b = _term_to_bin(t)
                if b is None:
                    continue
                # prefer first occurrence
                term_by_bin.setdefault(str(b), t)
            if baseline_bin not in term_by_bin:
                continue

            # Build covariance matrix for bin-terms
            C = pd.DataFrame(0.0, index=terms, columns=terms)
            for r in covg.itertuples(index=False):
                C.loc[r.term1, r.term2] = float(r.cov)
                C.loc[r.term2, r.term1] = float(r.cov)

            # Order bins by their end points
            gg2 = gg.copy()
            gg2["bin"] = gg2["bin"].astype(str)
            gg2["bin_end"] = gg2["bin"].map(_bin_end)
            gg2 = gg2[gg2["bin"].isin(term_by_bin.keys())].copy()
            if gg2.empty:
                continue
            gg2 = gg2.sort_values(["bin_end", "bin"], kind="mergesort")
            bins = gg2["bin"].tolist()
            if baseline_bin not in bins:
                bins.append(baseline_bin)
            bins = list(dict.fromkeys(bins))

            term_list = [term_by_bin[b] for b in bins]
            coef_list = [float(gg2.loc[gg2["bin"] == b, "coef"].iloc[0]) if b in set(gg2["bin"]) else float(gg.loc[gg["bin"].astype(str) == b, "coef"].iloc[0]) for b in bins]
            k = len(bins)
            base_idx = bins.index(baseline_bin)

            # A: (k-1) x k transform to relative-to-baseline
            keep_bins = [b for b in bins if b != baseline_bin]
            A = np.zeros((len(keep_bins), k), dtype=float)
            for i, b in enumerate(keep_bins):
                A[i, bins.index(b)] = 1.0
                A[i, base_idx] = -1.0
            Cov_bins = C.loc[term_list, term_list].to_numpy(dtype=float)
            Cov_rel = A @ Cov_bins @ A.T
            bvec = np.asarray(coef_list, dtype=float)
            b_rel = A @ bvec

            for i, b in enumerate(keep_bins):
                var = float(Cov_rel[i, i])
                se = math.sqrt(var) if np.isfinite(var) and var > 0 else float("nan")
                coef = float(b_rel[i])
                if se and np.isfinite(se) and se > 0 and np.isfinite(coef):
                    z = coef / se
                    p = float(2.0 * (1.0 - _norm_cdf(abs(z))))
                    ci_low = float(coef - 1.96 * se)
                    ci_high = float(coef + 1.96 * se)
                else:
                    p = float("nan")
                    ci_low = float("nan")
                    ci_high = float("nan")
                rel_rows.append(
                    {
                        "metric": metric,
                        "bin": b,
                        "baseline_bin": baseline_bin,
                        "coef": coef,
                        "se": se,
                        "p": p,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n_rows": int(gg2["n_rows"].max()) if "n_rows" in gg2.columns else int(gg["n_rows"].max()),
                        "n_threads": int(gg2["n_threads"].max()) if "n_threads" in gg2.columns else int(gg["n_threads"].max()),
                    }
                )

            # Save covariance (stacked) for the relative coefficients
            for i, b1 in enumerate(keep_bins):
                for j, b2 in enumerate(keep_bins):
                    rel_cov_rows.append(
                        {
                            "metric": metric,
                            "baseline_bin": baseline_bin,
                            "bin1": b1,
                            "bin2": b2,
                            "cov": float(Cov_rel[i, j]),
                        }
                    )

    rel_df = pd.DataFrame(rel_rows)
    rel_cov_df = pd.DataFrame(rel_cov_rows)
    if not rel_df.empty:
        rel_df["sample"] = "mature"
    if not rel_cov_df.empty:
        rel_cov_df["sample"] = "mature"
    save_parquet(rel_df, f"{ART}/event_coeffs_edge_main_rel.parquet")
    save_parquet(rel_cov_df, f"{ART}/event_cov_edge_main_rel.parquet")

    print(
        "Step05 complete. "
        f"Saved {len(agg):,} time-binned rows, {len(did_df):,} OLS rows, "
        f"{len(dr_mature_df):,} DR(mature) rows, {len(dr_post_df):,} DR(post-only) rows, "
        f"and {len(panel):,} edge-bin panel rows."
    )


if __name__ == "__main__":
    run()
