"""
revlib.py — Shared estimation core for the journal-revision analyses.

Self-contained re-implementation of the paper's `matched_dr_aipw` estimator
(cross-fitted matched doubly robust / AIPW ATT) that operates directly on the
aggregate artifacts produced by the main pipeline (``src/pipeline/step0*``).
It is used by the additional revision analyses in this folder and is validated
to reproduce ``artifacts/main_effects/att_post_main.parquet`` (Sample A
formation reciprocity ATT -0.185 vs. -0.187; SE 0.041 vs. 0.040; n exact).

Two estimators are provided:
  * ``matched_aipw_att`` — odds-augmented (Hajek self-normalized) AIPW with the
    weighted sandwich (efficient influence function) SE. Primary for Sample A.
  * ``matched_dr_att``   — outcome-adjusted matched DR without the propensity-
    odds term, with a multiplier bootstrap SE. Primary for the Sample B
    incumbent-difference outcomes, where the odds term is ill-conditioned.

Coding conventions (columns in the pipeline artifacts):
  * treated  <=> anchor_source == 'tstar'
  * control  <=> anchor_source == 'matched_median_latency'
  * matching weights / roles / propensity in matching/weights.parquet
  * matched treated->control pairs in matching/matched_pairs.parquet
  * pre-anchor covariates in threads/thread_meta.parquet
  * all outcomes (pre/post/diff, budget/time/time_inc variants) in outcomes/outcomes.parquet

Set the ``TSC_ART`` environment variable to point at the artifacts directory,
or place this folder at the repository root so ``../artifacts`` resolves.
"""
import os
import numpy as np
import pandas as pd

ART = os.environ.get("TSC_ART") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"
)

# ------------------------------------------------------------------ loaders

def _p(*parts):
    return os.path.join(ART, *parts)


def load_thread_meta():
    tm = pd.read_parquet(_p("threads", "thread_meta.parquet"))
    ca = pd.to_datetime(tm["created_at"], errors="coerce")
    tm["hour"] = ca.dt.hour.fillna(-1).astype(int)
    tm["dow"] = ca.dt.dayofweek.fillna(-1).astype(int)
    # device source -> integer code
    tm["source_code"] = tm["source"].astype("category").cat.codes
    return tm


def load_core(outcome_cols=None, extra_meta=True):
    """Return one row per analyzable thread with treatment, weight,
    propensity, pre-anchor covariates, and requested outcome columns."""
    keep_o = [
        "mblogid", "anchor_source", "sample_group",
        "n_pre_time_inc_edges", "n_post_time_inc_edges",
        "n_pre_time_edges", "n_post_time_edges",
        "n_pre_main_edges", "n_post_main_edges",
        "mature_time_ok", "budget_ok", "main_budget_ok",
        "age_at_anchor_min",
    ]
    if outcome_cols:
        keep_o = keep_o + [c for c in outcome_cols if c not in keep_o]
    o = pd.read_parquet(_p("outcomes", "outcomes.parquet"), columns=keep_o)
    o["T"] = (o["anchor_source"] == "tstar").astype(int)

    w = pd.read_parquet(_p("matching", "weights.parquet"))  # mblogid,w,role,n_pairs,propensity
    df = o.merge(w[["mblogid", "w", "role", "n_pairs", "propensity"]], on="mblogid", how="inner")

    if extra_meta:
        tm = load_thread_meta()
        cov = tm[["mblogid", "post_len", "post_has_question", "post_has_hashtag",
                  "pic_num", "source_code", "hour", "dow", "early_human_comments",
                  "op_id", "n_comments"]]
        df = df.merge(cov, on="mblogid", how="left")
    return df


COV_NUM = ["post_len", "post_has_question", "post_has_hashtag", "pic_num",
           "source_code", "hour", "dow", "early_human_comments", "propensity"]


def _xmat(df):
    X = df[COV_NUM].copy()
    for c in COV_NUM:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X.values.astype(float)


# ------------------------------------------------------- matched-pair table

def build_pairs(df, outcome, pair_path=None):
    """Attach a matched-pair table (treated t, control c) restricted to the
    analysis population (rows of `df` with a defined outcome), returning a
    tidy frame with per-pair residuals ready for the AIPW aggregation.

    Requires df to already contain columns: mblogid, T, outcome, and the
    cross-fitted mu0 prediction column 'mu0'.
    """
    if pair_path is None:
        pair_path = _p("matching", "matched_pairs.parquet")
    mp = pd.read_parquet(pair_path)  # mblogid_t, mblogid_c, dist

    sub = df[["mblogid", "T", outcome, "mu0", "w"]].dropna(subset=[outcome, "mu0"])
    treated = sub[sub["T"] == 1].set_index("mblogid")
    control = sub[sub["T"] == 0].set_index("mblogid")

    mp = mp[mp["mblogid_t"].isin(treated.index) & mp["mblogid_c"].isin(control.index)].copy()
    mp["yt"] = treated[outcome].reindex(mp["mblogid_t"]).values
    mp["mu0t"] = treated["mu0"].reindex(mp["mblogid_t"]).values
    mp["yc"] = control[outcome].reindex(mp["mblogid_c"]).values
    mp["mu0c"] = control["mu0"].reindex(mp["mblogid_c"]).values
    # per-treated number of matched controls (for weight sharing)
    kt = mp.groupby("mblogid_t")["mblogid_c"].transform("size")
    mp["pair_w"] = 1.0 / kt
    return mp, treated


# --------------------------------------------------------- outcome model

def crossfit_mu0(df, outcome, K=5, seed=0):
    """Cross-fitted E[Y | X, T=0] predicted for every unit (out-of-fold).
    Uses HistGradientBoostingRegressor to match the GBT flavor of the paper.
    Returns a positional numpy array aligned to df row order (NaN where the
    outcome is missing); index-safe under bootstrap resampling with dup indices.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    y_all = pd.to_numeric(df[outcome], errors="coerce").values.astype(float)
    obs = ~np.isnan(y_all)
    Xall = _xmat(df)
    T = df["T"].values
    out = np.full(len(df), np.nan)
    pos = np.where(obs)[0]
    X = Xall[pos]; y = y_all[pos]; Tp = T[pos]
    rng = np.random.RandomState(seed)
    folds = rng.randint(0, K, size=len(pos))
    for k in range(K):
        tr = (folds != k) & (Tp == 0)          # train on controls only
        te = folds == k
        if tr.sum() < 50:
            fill = y[(folds != k) & (Tp == 0)].mean() if ((folds != k) & (Tp == 0)).any() else y.mean()
            out[pos[te]] = fill
            continue
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=200,
                                          learning_rate=0.05, random_state=seed)
        m.fit(X[tr], y[tr])
        out[pos[te]] = m.predict(X[te])
    return out


# --------------------------------------------------------- AIPW ATT + BS

def matched_aipw_att(df, outcome, K=5, seed=0, clip=0.01, min_treated=25,
                     mu0=None):
    """Cross-fitted matched doubly-robust (AIPW) ATT with the weighted
    sandwich (efficient influence function) SE.

    Reconstructs the manuscript's `matched_dr_aipw` estimator. Validated
    against att_post_main.parquet: Sample A reciprocity att -0.185 (ref
    -0.187), se 0.0409 (ref 0.040), n_treated 31075, n_control 40007,
    ess_control 8135 -- all match.

    Estimator (Hajek self-normalized AIPW on the matched population):
        res_i = Y_i - mu0(X_i)                       (mu0 cross-fit on controls)
        ATT   = wmean_{T=1}(res; w) - wmean_{T=0}(res; (e/(1-e))*w)
        phi_i = T_i res_i - (1-T_i)(e_i/(1-e_i)) res_i - T_i*ATT
        SE    = sqrt( sum_i w_i^2 phi_i^2 ) / sum_i w_i T_i
    `df` must contain: T, w, propensity, and `outcome`. Pass a precomputed
    `mu0` array (aligned to df rows) to reuse across calls.
    """
    from scipy import stats
    d = df.copy()
    d["propensity"] = pd.to_numeric(d["propensity"], errors="coerce").clip(clip, 1 - clip)
    if mu0 is None:
        mu0 = crossfit_mu0(d, outcome, K=K, seed=seed)
    d = d.assign(_mu0=mu0)
    d = d.dropna(subset=[outcome, "_mu0", "propensity", "w"])
    T = d["T"].values.astype(float)
    w = d["w"].values.astype(float)
    e = d["propensity"].values.astype(float)
    res = (pd.to_numeric(d[outcome], errors="coerce").values - d["_mu0"].values)
    nT = int((T == 1).sum()); nC = int((T == 0).sum())
    if nT < min_treated or nC < min_treated:
        return dict(outcome=outcome, att=np.nan, se=np.nan, p=np.nan,
                    ci_low=np.nan, ci_high=np.nan, n_treated=nT, n_control=nC,
                    ess_control=np.nan, ctrl_mean=np.nan, note="insufficient")
    odds = e / (1 - e)
    wT = w[T == 1]; wC = w[T == 0]
    a = np.average(res[T == 1], weights=wT)
    cw = odds[T == 0] * wC
    b = np.average(res[T == 0], weights=cw)
    att = float(a - b)
    phi = T * res - (1 - T) * odds * res - T * att
    denom = (w * T).sum()
    se = float(np.sqrt(np.sum(w ** 2 * phi ** 2)) / denom)
    z = att / se if se > 0 else np.nan
    p = float(2 * stats.norm.sf(abs(z))) if se > 0 else np.nan
    ess_c = float(wC.sum() ** 2 / (wC ** 2).sum())
    ctrl_mean = float(np.average(pd.to_numeric(d[outcome], errors="coerce").values[T == 0], weights=cw))
    return dict(outcome=outcome, att=att, se=se, p=p,
                ci_low=att - 1.96 * se, ci_high=att + 1.96 * se,
                n_treated=nT, n_control=nC, ess_control=ess_c, ctrl_mean=ctrl_mean)


def matched_dr_att(df, outcome, K=5, seed=0, B=500, min_treated=25, mu0=None):
    """Stable bias-corrected matched DR ATT (Abadie--Imbens style): outcome-
    regression adjustment on the matched population with matching weights, but
    WITHOUT the propensity-odds IPW augmentation. Doubly robust (matching +
    outcome model). SE via independent treated/control multiplier bootstrap.

    Preferred for the Sample B incumbent-difference outcomes, where the odds
    augmentation is ill-conditioned on the degenerate small-graph reciprocity
    difference (controls reused many times x outcome in {-1,0,.5,1}). On the
    large bounded Sample A outcomes it agrees closely with matched_aipw_att.
    """
    from scipy import stats
    d = df.copy()
    if mu0 is None:
        mu0 = crossfit_mu0(d, outcome, K=K, seed=seed)
    d = d.assign(_mu0=mu0).dropna(subset=[outcome, "_mu0", "w"])
    T = (d["T"] == 1).values
    w = d["w"].values.astype(float)
    res = (pd.to_numeric(d[outcome], errors="coerce").values - d["_mu0"].values)
    nT = int(T.sum()); nC = int((~T).sum())
    if nT < min_treated or nC < min_treated:
        return dict(outcome=outcome, att=np.nan, se=np.nan, p=np.nan,
                    ci_low=np.nan, ci_high=np.nan, n_treated=nT, n_control=nC,
                    estimator="matched_dr", note="insufficient")
    resT, wT = res[T], w[T]; resC, wC = res[~T], w[~T]
    att = float(np.average(resT, weights=wT) - np.average(resC, weights=wC))
    rng = np.random.RandomState(seed + 7)
    boot = np.empty(B)
    for b in range(B):
        mT = rng.exponential(1.0, size=wT.size)
        mC = rng.exponential(1.0, size=wC.size)
        boot[b] = np.average(resT, weights=mT * wT) - np.average(resC, weights=mC * wC)
    se = float(np.std(boot, ddof=1))
    z = att / se if se > 0 else np.nan
    p = float(2 * stats.norm.sf(abs(z))) if se > 0 else np.nan
    return dict(outcome=outcome, att=att, se=se, p=p, ci_low=att - 1.96 * se,
                ci_high=att + 1.96 * se, n_treated=nT, n_control=nC,
                estimator="matched_dr")


def all_estimators(df, outcome, K=5, seed=0):
    """Run naive, stable matched-DR, and paper-exact odds-AIPW on one outcome,
    reusing a single cross-fitted mu0. Returns a tidy 3-row DataFrame."""
    d = df.copy()
    d["propensity"] = pd.to_numeric(d["propensity"], errors="coerce").clip(0.01, 0.99)
    mu0 = crossfit_mu0(d, outcome, K=K, seed=seed)
    nd, _ = naive_matched_diff(d, outcome)
    r_dr = matched_dr_att(d, outcome, mu0=mu0, seed=seed)
    r_aipw = matched_aipw_att(d, outcome, mu0=mu0, seed=seed)
    rows = [dict(estimator="naive_matched", outcome=outcome, att=nd, se=np.nan, p=np.nan,
                 n_treated=r_dr["n_treated"], n_control=r_dr["n_control"]),
            dict(estimator="matched_dr", outcome=outcome, att=r_dr["att"], se=r_dr["se"],
                 p=r_dr["p"], n_treated=r_dr["n_treated"], n_control=r_dr["n_control"]),
            dict(estimator="odds_aipw", outcome=outcome, att=r_aipw["att"], se=r_aipw["se"],
                 p=r_aipw["p"], n_treated=r_aipw["n_treated"], n_control=r_aipw["n_control"])]
    return pd.DataFrame(rows)


def naive_matched_diff(df, outcome, pair_path=None):
    """Matching-weighted mean difference (no bias correction) — sanity check
    that should reproduce matched_effects.parquet diffs."""
    df = df.copy()
    df["mu0"] = 0.0
    mp, _ = build_pairs(df, outcome, pair_path=pair_path)
    mp["res_c"] = mp["pair_w"] * mp["yc"]
    grp = mp.groupby("mblogid_t")
    tau_g = grp["yt"].first() - grp["res_c"].sum()
    return float(tau_g.mean()), int(len(tau_g))
