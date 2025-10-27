import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

from ..models.prosocial import ProsocialAnalyzer
from ..utils.io_utils import iter_json_like, ensure_dir


def load_comments(path: Path, text_col: str = "content") -> pd.DataFrame:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = list(iter_json_like(path.as_posix()))
        df = pd.DataFrame(rows)
    elif path.suffix.lower() == ".json":
        df = pd.read_json(path, orient="records")
    else:
        df = pd.read_parquet(path)
    if text_col not in df.columns:
        raise ValueError(f"{text_col} not found in {path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Export a sample of tone classifier predictions for manual review.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--comments",
        type=Path,
        default=Path("artifacts/ingested/comments.parquet"),
        help="Path to comments data (parquet/json/jsonl).",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/tone/tone_audit.csv"))
    parser.add_argument("--text-col", type=str, default="content")
    parser.add_argument("--n-positive", type=int, default=200, help="Number of predicted positives to sample.")
    parser.add_argument("--n-negative", type=int, default=200, help="Number of predicted negatives to sample.")
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    comments = load_comments(args.comments, text_col=args.text_col)
    analyzer = ProsocialAnalyzer.from_config(cfg)
    if analyzer.classifier is None or analyzer.vectorizer is None:
        raise RuntimeError("Tone classifier not available. Train it first (see train_tone_classifier.py).")

    annotations = analyzer.annotate(comments[args.text_col])
    df = pd.DataFrame(
        {
            "content": comments[args.text_col].fillna(""),
            "score": annotations["score"],
            "pred": annotations["pred"],
        }
    )

    rng = np.random.default_rng(args.seed)
    samples = []
    if args.n_positive > 0:
        pos = df[df["pred"] == 1]
        n = min(len(pos), args.n_positive)
        if n > 0:
            samples.append(pos.sample(n=n, random_state=args.seed).assign(sample_label="positive"))
    if args.n_negative > 0:
        neg = df[df["pred"] == 0]
        n = min(len(neg), args.n_negative)
        if n > 0:
            samples.append(neg.sample(n=n, random_state=args.seed + 1).assign(sample_label="negative"))

    if not samples:
        raise RuntimeError("No samples available with the requested counts; adjust n_positive/n_negative.")

    audit = pd.concat(samples, ignore_index=True)
    ensure_dir(args.output.parent.as_posix())
    audit.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(audit)} audit rows to {args.output}")
    print("Columns: content, score, pred, sample_label")


if __name__ == "__main__":
    main()
