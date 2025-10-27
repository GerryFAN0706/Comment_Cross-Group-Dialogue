import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..utils.io_utils import ensure_dir
from ..utils.text_utils import contains_any


def _flatten_lexicon(lexicon: Dict[str, List[str]]) -> List[str]:
    terms: List[str] = []
    for values in lexicon.values():
        if not values:
            continue
        terms.extend([v for v in values if v])
    # preserve order while dropping duplicates
    seen = set()
    unique_terms = []
    for term in terms:
        if term not in seen:
            unique_terms.append(term)
            seen.add(term)
    return unique_terms


def prosocial_flags(text: str, lexicon: Dict[str, List[str]]) -> Dict[str, int]:
    text = text or ""
    return {
        "gratitude": int(contains_any(text, lexicon.get("gratitude", []))),
        "support": int(contains_any(text, lexicon.get("support", []))),
        "deescalation": int(contains_any(text, lexicon.get("deescalation", []))),
        "emoji_support": int(contains_any(text, lexicon.get("emojis", []))),
    }


@dataclass
class ProsocialAnalyzer:
    lexicon: Dict[str, List[str]]
    classifier: Optional[LogisticRegression] = None
    vectorizer: Optional[TfidfVectorizer] = None
    threshold: float = 0.5
    metadata: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        cfg: Dict,
        comments: Optional[pd.DataFrame] = None,
        text_col: str = "content",
    ) -> "ProsocialAnalyzer":
        lexicon = cfg.get("prosocial_lexicon", {}) or {}
        clf_cfg = cfg.get("tone_classifier", {})
        classifier = None
        vectorizer = None
        threshold = clf_cfg.get("threshold", 0.5)
        metadata: Dict[str, float] = {}

        if clf_cfg.get("enabled"):
            model_path = clf_cfg.get("model_path")
            payload = None
            if model_path and os.path.exists(model_path):
                payload = joblib.load(model_path)
            elif model_path and comments is not None:
                payload = _train_prosocial_classifier(
                    comments,
                    text_col=text_col,
                    lexicon=lexicon,
                    clf_cfg=clf_cfg,
                )
                if payload:
                    ensure_dir(os.path.dirname(model_path))
                    joblib.dump(payload, model_path)
            if payload:
                classifier = payload.get("classifier")
                vectorizer = payload.get("vectorizer")
                threshold = payload.get("threshold", threshold)
                metadata = payload.get("metadata", {})

        return cls(lexicon=lexicon, classifier=classifier, vectorizer=vectorizer, threshold=threshold, metadata=metadata)

    def annotate(self, texts: pd.Series) -> pd.DataFrame:
        if texts is None or texts.empty:
            return pd.DataFrame(columns=["gratitude", "support", "deescalation", "emoji_support", "score", "pred"])
        series = texts.fillna("").astype(str)
        flag_rows = [prosocial_flags(t, self.lexicon) for t in series]
        flags_df = pd.DataFrame(flag_rows)

        if self.classifier is not None and self.vectorizer is not None and not flags_df.empty:
            X = self.vectorizer.transform(series)
            scores = self.classifier.predict_proba(X)[:, 1]
        else:
            scores = flags_df.max(axis=1).astype(float).values if not flags_df.empty else np.array([])

        annotations = flags_df.copy()
        annotations["score"] = scores
        annotations["pred"] = (scores >= self.threshold).astype(int) if len(scores) else []
        return annotations

    def aggregate(self, df: Optional[pd.DataFrame], text_col: str = "content") -> Dict[str, float]:
        if df is None or df.empty:
            return {k: np.nan for k in [
                "gratitude",
                "support",
                "deescalation",
                "emoji_support",
                "prosocial_any",
                "prosocial_score",
            ]}
        annotations = self.annotate(df[text_col])
        if annotations.empty:
            return {k: np.nan for k in [
                "gratitude",
                "support",
                "deescalation",
                "emoji_support",
                "prosocial_any",
                "prosocial_score",
            ]}
        stats = {col: float(annotations[col].mean()) for col in ["gratitude", "support", "deescalation", "emoji_support"]}
        stats["prosocial_any"] = float(annotations["pred"].mean()) if "pred" in annotations else np.nan
        stats["prosocial_score"] = float(annotations["score"].mean()) if "score" in annotations else np.nan
        return stats


def _train_prosocial_classifier(
    comments: pd.DataFrame,
    text_col: str,
    lexicon: Dict[str, List[str]],
    clf_cfg: Dict,
) -> Optional[Dict]:
    texts = comments[text_col].fillna("").astype(str)
    lex_terms = _flatten_lexicon(lexicon)
    if not lex_terms:
        return None

    mask_positive = texts.apply(lambda t: contains_any(t, lex_terms))
    positives = texts[mask_positive]
    min_positive = clf_cfg.get("min_positive", 200)
    if len(positives) < min_positive:
        return None

    negatives = texts[~mask_positive]
    if negatives.empty:
        return None

    seed = clf_cfg.get("seed", 2025)
    max_positive = min(clf_cfg.get("max_positive", len(positives)), len(positives))
    max_negative = min(clf_cfg.get("max_negative", len(negatives)), len(negatives))

    pos_sample = positives.sample(n=max_positive, random_state=seed, replace=False)
    neg_sample = negatives.sample(n=max_negative, random_state=seed, replace=False)

    train_texts = list(pos_sample.values) + list(neg_sample.values)
    train_labels = np.array([1] * len(pos_sample) + [0] * len(neg_sample))

    vectorizer_cfg = clf_cfg.get("vectorizer", {})
    vectorizer = TfidfVectorizer(
        analyzer=vectorizer_cfg.get("analyzer", "char"),
        ngram_range=tuple(vectorizer_cfg.get("ngram_range", [2, 4])),
        min_df=vectorizer_cfg.get("min_df", 2),
    )
    X = vectorizer.fit_transform(train_texts)

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="liblinear",
    )
    classifier.fit(X, train_labels)

    payload = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "threshold": clf_cfg.get("threshold", 0.5),
        "metadata": {
            "n_positive": int(len(pos_sample)),
            "n_negative": int(len(neg_sample)),
        },
    }
    return payload
