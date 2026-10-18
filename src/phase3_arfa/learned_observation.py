"""Observation-only control trained on Phase 1.5 development labels."""
from __future__ import annotations

from pathlib import Path

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.common.utils import read_jsonl
from src.phase1_5_shadow.observation_baseline import _structured_features


class LearnedObservationScorer:
    def __init__(self, annotation_path: str, source_map_path: str, seed: int = 1507) -> None:
        sources = {row["annotation_id"]: row for row in read_jsonl(source_map_path)}
        rows = [
            row for row in read_jsonl(annotation_path)
            if sources.get(row["annotation_id"], {}).get("split") == "shadow_development"
            and row.get("slow_reasoning_needed") in {"yes", "no"}
        ]
        if not rows:
            raise ValueError("No binary shadow_development labels for learned observation-only control")
        labels = [int(row["slow_reasoning_needed"] == "yes") for row in rows]
        if len(set(labels)) != 2:
            raise ValueError("Learned observation-only control requires both label classes")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        text_features = self.vectorizer.fit_transform(
            [str(row.get("actual_observation") or "<empty observation>") for row in rows]
        )
        structured = csr_matrix([_structured_features(row) for row in rows])
        self.classifier = LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=seed, solver="liblinear"
        )
        self.classifier.fit(hstack([text_features, structured], format="csr"), labels)
        self.training_summary = {
            "training_split": "shadow_development",
            "training_items": len(rows),
            "positive_items": sum(labels),
            "annotation_path": str(Path(annotation_path)),
            "source_map_path": str(Path(source_map_path)),
            "text_backend": "tfidf",
            "seed": seed,
        }

    def score(self, observation: str, exit_code: int) -> float:
        # The same parser/features are used offline and online; no action or expectation enters.
        row = {"actual_observation": observation, "exit_code": exit_code}
        features = hstack(
            [
                self.vectorizer.transform([observation or "<empty observation>"]),
                csr_matrix([_structured_features(row)]),
            ],
            format="csr",
        )
        return float(self.classifier.predict_proba(features)[0, list(self.classifier.classes_).index(1)])
