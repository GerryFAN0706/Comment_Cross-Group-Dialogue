import argparse
from pathlib import Path
import yaml
import pandas as pd

from ..models.prosocial import ProsocialAnalyzer
from ..utils.io_utils import iter_json_like, ensure_dir


def load_comments(path: Path, sample_frac: float | None = None, seed: int = 2025) -> pd.DataFrame:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = list(iter_json_like(path.as_posix()))
        df = pd.DataFrame(rows)
    elif path.suffix.lower() == ".json":
        df = pd.read_json(path, orient="records")
    else:
        df = pd.read_parquet(path)
    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=seed)
    return df


def main():
    parser = argparse.ArgumentParser(description="Train / refresh the prosocial tone classifier.")
    parser.add_argument(
        "--config",
        default=Path("config.yaml"),
        type=Path,
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--comments",
        default=Path("artifacts/ingested/comments.parquet"),
        type=Path,
        help="Path to comments data (parquet/json/jsonl). Defaults to ingested parquet.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional fraction of comments to sample for quicker training (e.g., 0.3).",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore any existing saved classifier and retrain from scratch.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    clf_cfg = cfg.get("tone_classifier", {})
    model_path = Path(clf_cfg.get("model_path", "artifacts/tone/prosocial_classifier.joblib"))
    if args.force_retrain and model_path.exists():
        model_path.unlink()

    print(f"Loading comments from {args.comments} ...")
    comments = load_comments(args.comments, sample_frac=args.sample_frac, seed=clf_cfg.get("seed", 2025))
    print(f"Loaded {len(comments):,} comments.")

    analyzer = ProsocialAnalyzer.from_config(cfg, comments)
    if analyzer.classifier is None:
        raise RuntimeError("Classifier training failed - ensure lexicon has sufficient coverage or adjust thresholds.")

    ensure_dir(model_path.parent.as_posix())
    payload = {
        "vectorizer": analyzer.vectorizer,
        "classifier": analyzer.classifier,
        "threshold": analyzer.threshold,
        "metadata": analyzer.metadata,
    }
    import joblib

    joblib.dump(payload, model_path)
    print(f"Saved classifier to {model_path}")
    print("Metadata:", analyzer.metadata)


if __name__ == "__main__":
    main()
