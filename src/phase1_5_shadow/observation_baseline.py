from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ERROR_PATTERN = re.compile(
    r"\b(error|failed|failure|exception|traceback|syntaxerror|not found|permission denied|command not found)\b",
    flags=re.IGNORECASE,
)
SUCCESS_PATTERN = re.compile(r"\b(passed|success|successful|ok|completed)\b", flags=re.IGNORECASE)


def _label(row: dict[str, Any]) -> int | None:
    return {"yes": 1, "no": 0}.get(str(row.get("slow_reasoning_needed", "")).strip().lower())


def _text(row: dict[str, Any]) -> str:
    return str(row.get("actual_observation") or "<empty observation>")


def _structured_features(row: dict[str, Any]) -> list[float]:
    text = _text(row)
    lower = text.lower()
    stderr_match = re.search(r"(?:^|\n)stderr:\n(.+)$", text, flags=re.DOTALL)
    stdout_match = re.search(r"(?:^|\n)stdout:\n(.*?)(?:\nstderr:\n|$)", text, flags=re.DOTALL)
    stderr = stderr_match.group(1).strip() if stderr_match else ""
    stdout = stdout_match.group(1).strip() if stdout_match else ""
    return [
        float(int(row.get("exit_code") or 0) != 0),
        float(bool(stderr)),
        float(not stdout),
        float(not text.strip()),
        float(bool(ERROR_PATTERN.search(lower))),
        float(bool(SUCCESS_PATTERN.search(lower))),
        math.log1p(len(text)) / 10.0,
        math.log1p(text.count("\n") + 1) / 5.0,
    ]


def fit_observation_only_scores(
    rows: list[dict[str, Any]],
    model_name: str,
    require_model: bool,
    seed: int,
) -> tuple[list[float], dict[str, Any]]:
    development_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("split") == "shadow_development" and _label(row) is not None
    ]
    labels = np.array([_label(rows[index]) for index in development_indices], dtype=int)
    if len(development_indices) < 2 or len(set(labels.tolist())) < 2:
        raise ValueError("Learned observation-only baseline requires both binary classes in shadow_development")

    texts = [_text(row) for row in rows]
    development_texts = [texts[index] for index in development_indices]
    backend = model_name
    if model_name.lower() in {"tfidf", "tfidf_fallback"}:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        vectorizer.fit(development_texts)
        text_features = vectorizer.transform(texts)
        feature_count = len(vectorizer.vocabulary_)
        backend = "tfidf_fallback"
    else:
        try:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(model_name)
            text_features = csr_matrix(encoder.encode(texts, normalize_embeddings=True))
            feature_count = int(text_features.shape[1])
        except Exception as exc:
            if require_model:
                raise RuntimeError(f"Observation-only embedding model '{model_name}' is unavailable") from exc
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
            vectorizer.fit(development_texts)
            text_features = vectorizer.transform(texts)
            feature_count = len(vectorizer.vocabulary_)
            backend = "tfidf_fallback"

    structured = csr_matrix(np.asarray([_structured_features(row) for row in rows], dtype=float))
    features = hstack([text_features, structured], format="csr")
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
        solver="liblinear",
    )
    classifier.fit(features[development_indices], labels)
    scores = classifier.predict_proba(features)[:, list(classifier.classes_).index(1)]
    return scores.astype(float).tolist(), {
        "method": "logistic_regression",
        "training_split": "shadow_development",
        "training_binary_items": len(development_indices),
        "training_positive_items": int(np.sum(labels == 1)),
        "training_negative_items": int(np.sum(labels == 0)),
        "text_backend": backend,
        "text_feature_count": feature_count,
        "structured_feature_count": structured.shape[1],
        "allowed_inputs": ["actual_observation", "exit_code", "stderr_derived_from_actual_observation"],
        "forbidden_inputs": ["expectation", "plan", "action", "continuation"],
        "random_seed": seed,
    }
