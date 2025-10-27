import yaml
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
from ..utils.io_utils import ensure_dir, save_parquet

SUMMARY_DIR = Path("artifacts/sample_summary")
FIG_DIR = Path("artifacts/figures/sample")

EXPECTED_FILES = {
    "outcomes": "artifacts/outcomes/outcomes.parquet",
    "matching_diag": "artifacts/matching/diagnostics.parquet",
    "did_results": "artifacts/event_study/did_results.parquet",
    "first_stage": "artifacts/instrument/first_stage_summary.parquet",
    "twosri": "artifacts/instrument/twosri_results.parquet",
    "robust_excl": "artifacts/robustness/exclude_mega_threads.parquet",
    "robust_placebo": "artifacts/robustness/placebo_summary.parquet",
    "balance": "artifacts/matching/balance_table.parquet",
    "matched_effects": "artifacts/matching/matched_effects.parquet",
    "event_summary": "artifacts/event_study/event_summary.parquet",
}


def _load(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception as exc:  # pragma: no cover
            print(f"Warning: failed to read {path}: {exc}")
    return None


def _plot_delta_bars(delta_series: pd.Series):
    ensure_dir(str(FIG_DIR))
    if delta_series.empty:
        plt.figure(figsize=(4, 3))
        plt.text(0.5, 0.5, "No treated threads in sample", ha="center", va="center", fontsize=10)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "delta_means.png", dpi=200)
        plt.close()
        return
    ax = delta_series.sort_values().plot(kind="barh", figsize=(6, 4), color="#4C72B0")
    ax.set_xlabel("Mean (post - pre)")
    ax.set_title("Thread-Level Outcome Shifts (Sample)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "delta_means.png", dpi=200)
    plt.close()


def _plot_did_results(df: pd.DataFrame):
    ensure_dir(str(FIG_DIR))
    if df is None or df.empty:
        plt.figure(figsize=(4, 3))
        plt.text(0.5, 0.5, "DiD skipped (insufficient treated)", ha="center", va="center", fontsize=10)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "did_coefs.png", dpi=200)
        plt.close()
        return
    sorted_df = df.sort_values("coef_treated")
    y_pos = range(len(sorted_df))
    plt.figure(figsize=(6, 4))
    plt.barh(y_pos, sorted_df["coef_treated"], xerr=1.96 * sorted_df["se"].fillna(0), color="#55A868", alpha=0.8)
    plt.yticks(y_pos, sorted_df["outcome"])
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("DiD Coefficient")
    plt.title("Sample DiD Estimates (treated effect)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "did_coefs.png", dpi=200)
    plt.close()


def _plot_event_series(summary: pd.DataFrame):
    ensure_dir(str(FIG_DIR))
    if summary is None or summary.empty:
        plt.figure(figsize=(4, 3))
        plt.text(0.5, 0.5, "Event summary unavailable", ha="center", va="center", fontsize=10)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "event_edges.png", dpi=200)
        plt.savefig(FIG_DIR / "event_root_share.png", dpi=200)
        plt.close("all")
        return
    summary = summary.copy()
    summary["bin_label"] = summary["bin"].astype(str)
    pivot_edges = summary.pivot(index="bin_label", columns="treated", values="mean_edges")
    pivot_root = summary.pivot(index="bin_label", columns="treated", values="mean_root_reply_rate")

    def _plot_line(pivot, fname, ylabel):
        plt.figure(figsize=(6, 4))
        if False in pivot.columns:
            plt.plot(pivot.index, pivot[False], marker="o", label="Control")
        if True in pivot.columns:
            plt.plot(pivot.index, pivot[True], marker="o", label="Treated")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Event bin (minutes relative to t*)")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / fname, dpi=200)
        plt.close()

    _plot_line(pivot_edges.fillna(0), "event_edges.png", "Mean edges per thread")
    _plot_line(pivot_root, "event_root_share.png", "Mean root-reply rate")


def _plot_bridge_distribution(outcomes: pd.DataFrame):
    ensure_dir(str(FIG_DIR))
    if outcomes is None or outcomes.empty or "all_DCBI_gender" not in outcomes.columns:
        plt.figure(figsize=(4, 3))
        plt.text(0.5, 0.5, "DC-BI distribution unavailable", ha="center", va="center", fontsize=10)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "dcbi_hist.png", dpi=200)
        plt.close()
        return
    treated = outcomes[outcomes["tstar"].notna()]["all_DCBI_gender"].dropna()
    control = outcomes[outcomes["tstar"].isna()]["all_DCBI_gender"].dropna()
    if len(treated) < 5 or len(control) < 5:
        plt.figure(figsize=(4, 3))
        plt.text(0.5, 0.5, "Insufficient DC-BI samples", ha="center", va="center", fontsize=10)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "dcbi_hist.png", dpi=200)
        plt.close()
        return
    plt.figure(figsize=(6, 4))
    bins = 30
    plt.hist(control, bins=bins, alpha=0.6, label="Control", color="#4C72B0")
    plt.hist(treated, bins=bins, alpha=0.6, label="Treated", color="#55A868")
    plt.xlabel("All-thread DC-BI (gender)")
    plt.ylabel("Count")
    plt.title("DC-BI Distribution (Treated vs Control)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dcbi_hist.png", dpi=200)
    plt.close()


def run():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    ensure_dir(str(SUMMARY_DIR))
    ensure_dir("artifacts/figures")
    summary_rows = []

    tone_cfg = cfg.get("tone_classifier", {})
    if tone_cfg.get("enabled"):
        model_path = Path(tone_cfg.get("model_path", ""))
        if model_path.exists():
            try:
                payload = joblib.load(model_path)
                threshold = payload.get("threshold", tone_cfg.get("threshold", 0.5))
                summary_rows.append({"metric": "tone_threshold", "value": threshold, "section": "tone_classifier"})
                metadata = payload.get("metadata") or {}
                for key, val in metadata.items():
                    summary_rows.append({"metric": f"tone_{key}", "value": val, "section": "tone_classifier"})
            except Exception as exc:  # pragma: no cover
                print(f"Warning: failed to load tone classifier: {exc}")

    outcomes = _load(EXPECTED_FILES["outcomes"])
    _plot_bridge_distribution(outcomes)
    if outcomes is not None and not outcomes.empty:
        delta_means = {}
        pre_cols = [c for c in outcomes.columns if c.startswith("pre_")]
        for pre_col in pre_cols:
            metric = pre_col.replace("pre_", "", 1)
            post_col = f"post_{metric}"
            if post_col in outcomes.columns:
                diff = (outcomes[post_col] - outcomes[pre_col]).dropna()
                if not diff.empty:
                    delta_means[f"d_{metric}"] = diff.mean()
        delta_series = pd.Series(delta_means)
        summary_rows.extend(
            {"metric": metric, "value": val, "section": "delta_mean"} for metric, val in delta_means.items()
        )
        summary_rows.append({"metric": "n_threads", "value": len(outcomes), "section": "counts"})
        summary_rows.append(
            {"metric": "share_treated", "value": outcomes["tstar"].notna().mean(), "section": "counts"}
        )
        _plot_delta_bars(delta_series)

    matching_diag = _load(EXPECTED_FILES["matching_diag"])
    if matching_diag is not None and not matching_diag.empty:
        for row in matching_diag.itertuples(index=False):
            summary_rows.append({"metric": row.metric, "value": row.value, "section": "matching"})

    balance = _load(EXPECTED_FILES["balance"])
    if balance is not None and not balance.empty:
        summary_rows.append(
            {
                "metric": "balance_max_abs_before",
                "value": float(balance["std_diff_before"].abs().max(skipna=True)),
                "section": "matching",
            }
        )
        if "std_diff_after" in balance.columns:
            summary_rows.append(
                {
                    "metric": "balance_max_abs_after",
                    "value": float(balance["std_diff_after"].abs().max(skipna=True)),
                    "section": "matching",
                }
            )

    did_results = _load(EXPECTED_FILES["did_results"])
    _plot_did_results(did_results)
    if did_results is not None and not did_results.empty:
        for row in did_results.itertuples(index=False):
            summary_rows.append(
                {
                    "metric": f"{row.outcome}_coef",
                    "value": row.coef_treated,
                    "section": "did",
                }
            )
            summary_rows.append(
                {
                    "metric": f"{row.outcome}_p",
                    "value": row.p,
                    "section": "did",
                }
            )

    first_stage = _load(EXPECTED_FILES["first_stage"])
    if first_stage is not None and not first_stage.empty:
        for row in first_stage.itertuples(index=False):
            summary_rows.append({"metric": f"first_stage_{row.param}", "value": row.value, "section": "instrument"})

    twosri = _load(EXPECTED_FILES["twosri"])
    if twosri is not None and not twosri.empty:
        for row in twosri.itertuples(index=False):
            summary_rows.append(
                {"metric": f"{row.outcome}_2sri_coef", "value": row.coef_treated, "section": "instrument"}
            )

    robust_excl = _load(EXPECTED_FILES["robust_excl"])
    if robust_excl is not None and not robust_excl.empty:
        for row in robust_excl.itertuples(index=False):
            summary_rows.append(
                {"metric": f"{row.metric}_exclude_top1", "value": row.exclude_top1, "section": "robustness"}
            )

    placebo = _load(EXPECTED_FILES["robust_placebo"])
    if placebo is not None and not placebo.empty:
        for row in placebo.itertuples(index=False):
            summary_rows.append(
                {"metric": f"{row.metric}_placebo", "value": row.placebo, "section": "robustness"}
            )

    matched_effects = _load(EXPECTED_FILES["matched_effects"])
    if matched_effects is not None and not matched_effects.empty:
        for row in matched_effects.itertuples(index=False):
            summary_rows.append(
                {
                    "metric": f"{row.metric}_diff_mean",
                    "value": row.diff_mean,
                    "section": "matched_effects",
                }
            )
            summary_rows.append(
                {
                    "metric": f"{row.metric}_diff_se",
                    "value": row.diff_se,
                    "section": "matched_effects",
                }
            )

    event_summary = _load(EXPECTED_FILES["event_summary"])
    _plot_event_series(event_summary)

    summary_df = pd.DataFrame(summary_rows)
    save_parquet(summary_df, SUMMARY_DIR / "sample_summary.parquet")
    print(f"Sample summary saved with {len(summary_df)} entries. Figures stored under {FIG_DIR}.")


if __name__ == "__main__":
    run()
