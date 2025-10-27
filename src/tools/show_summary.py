import argparse
from pathlib import Path
import pandas as pd


SUMMARY_PATH = Path("artifacts/sample_summary/sample_summary.parquet")
OUTCOMES_PATH = Path("artifacts/outcomes/outcomes.parquet")
BALANCE_PATH = Path("artifacts/matching/balance_table.parquet")
MATCHED_EFFECTS_PATH = Path("artifacts/matching/matched_effects.parquet")
EVENT_SUMMARY_PATH = Path("artifacts/event_study/event_summary.parquet")
TONE_AUDIT_PATH = Path("artifacts/tone/tone_audit_full.csv")


def main():
    parser = argparse.ArgumentParser(description="Print headline metrics after a pipeline run.")
    parser.add_argument("--summary", default=SUMMARY_PATH, type=Path)
    parser.add_argument("--outcomes", default=OUTCOMES_PATH, type=Path)
    parser.add_argument("--balance", default=BALANCE_PATH, type=Path)
    parser.add_argument("--matched-effects", default=MATCHED_EFFECTS_PATH, type=Path)
    parser.add_argument("--event-summary", default=EVENT_SUMMARY_PATH, type=Path)
    parser.add_argument("--tone-audit", default=TONE_AUDIT_PATH, type=Path)
    args = parser.parse_args()

    def load_parquet(path: Path) -> pd.DataFrame | None:
        if path.exists():
            return pd.read_parquet(path)
        return None

    summary = load_parquet(args.summary)
    outcomes = load_parquet(args.outcomes)
    balance = load_parquet(args.balance)
    matched = load_parquet(args.matched_effects)
    event_summary = load_parquet(args.event_summary)

    print("=== CommentR Sample Summary ===")
    if summary is not None and not summary.empty:
        headline = summary[summary["metric"].isin([
            "n_threads",
            "share_treated",
            "tone_threshold",
            "tone_n_positive",
            "tone_n_negative",
            "balance_max_abs_before",
            "balance_max_abs_after",
        ])]
        print(headline.to_string(index=False))
    else:
        print("No sample_summary.parquet found.")

    print("\n=== Outcome deltas (treated vs control) ===")
    if outcomes is not None and not outcomes.empty:
        treated_share = outcomes["tstar"].notna().mean()
        print(f"Treated threads: {outcomes['tstar'].notna().sum():,} ({treated_share:.2%})")
        delta_cols = [c for c in outcomes.columns if c.startswith("d_")]
        if delta_cols:
            print(outcomes[delta_cols].describe().T[["mean", "std", "min", "max"]])
        else:
            print("No delta columns available.")
    else:
        print("Outcomes parquet missing or empty.")

    print("\n=== Matching balance (std diff) ===")
    if balance is not None and not balance.empty:
        cols = ["feature", "std_diff_before", "std_diff_after"]
        existing = [c for c in cols if c in balance.columns]
        print(balance[existing].head(10).to_string(index=False))
    else:
        print("Balance table not available.")

    print("\n=== Matched effects summary ===")
    if matched is not None and not matched.empty:
        print(matched.to_string(index=False))
    else:
        print("Matched effects table not available.")

    print("\n=== Event-study aggregates ===")
    if event_summary is not None and not event_summary.empty:
        print(event_summary.head(10).to_string(index=False))
    else:
        print("Event summary table not available.")

    if args.tone_audit.exists():
        audit = pd.read_csv(args.tone_audit)
        total = len(audit)
        positives = (audit["sample_label"] == "positive").sum()
        negatives = (audit["sample_label"] == "negative").sum()
        print("\n=== Tone audit sample ===")
        print(f"Total rows: {total:,}  positives: {positives:,}  negatives: {negatives:,}")
    else:
        print("\nTone audit CSV not found. Run sample_tone_predictions to create one.")


if __name__ == "__main__":
    main()
