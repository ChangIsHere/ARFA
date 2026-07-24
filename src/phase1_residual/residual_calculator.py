from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.common.config import load_simple_yaml
from src.common.utils import read_jsonl, write_csv


ERROR_PATTERNS = [
    r"\bfailed\b",
    r"\bfailures?\b",
    r"\berror\b",
    r"\bexception\b",
    r"traceback",
    r"syntaxerror",
    r"modulenotfounderror",
    r"no such file",
    r"not found",
    r"permission denied",
    r"command not found",
]

SUCCESS_PATTERNS = [
    r"\bpassed\b",
    r"all checks passed",
    r"\bok\b",
    r"success",
    r"compiled",
]


def _try_sentence_transformer(expectations: list[str], observations: list[str], model_name: str) -> tuple[list[float], str]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(str(exc)) from exc

    model = SentenceTransformer(model_name)
    exp_embeddings = model.encode(expectations, normalize_embeddings=True)
    obs_embeddings = model.encode(observations, normalize_embeddings=True)
    scores = []
    for exp_vec, obs_vec in zip(exp_embeddings, obs_embeddings):
        sim = float(np.dot(exp_vec, obs_vec))
        scores.append(float(max(0.0, min(2.0, 1.0 - sim))))
    return scores, model_name


def _tfidf_residual(expectations: list[str], observations: list[str]) -> list[float]:
    corpus = expectations + observations
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", lowercase=True)
    matrix = vectorizer.fit_transform(corpus)
    exp_matrix = matrix[: len(expectations)]
    obs_matrix = matrix[len(expectations) :]
    similarities = cosine_similarity(exp_matrix, obs_matrix).diagonal()
    return [float(max(0.0, min(1.0, 1.0 - value))) for value in similarities]


def _expectation(record: dict[str, Any]) -> str:
    return str(record.get("expected_outcome") or record.get("expectation") or "")


def _observation(record: dict[str, Any]) -> str:
    return str(record.get("actual_observation") or record.get("observation") or "")


def _label(record: dict[str, Any]) -> str:
    return str(record.get("reasoning_needed", "")).lower()


def tfidf_residuals(records: list[dict[str, Any]]) -> list[float]:
    expectations = [_expectation(record) for record in records]
    observations = [_observation(record) or "<empty output>" for record in records]
    return _tfidf_residual(expectations, observations)


def embedding_residuals(records: list[dict[str, Any]], model_name: str, require_model: bool = False) -> tuple[list[float], str]:
    expectations = [_expectation(record) for record in records]
    observations = [_observation(record) or "<empty output>" for record in records]
    try:
        return _try_sentence_transformer(expectations, observations, model_name)
    except Exception as exc:
        if require_model:
            raise RuntimeError(
                f"Embedding model '{model_name}' is unavailable. Install sentence-transformers and cache the model before final Phase 1."
            ) from exc
        return _tfidf_residual(expectations, observations), "tfidf_fallback"


def rule_residual(record: dict[str, Any]) -> float:
    observation = _observation(record).lower()
    expectation = _expectation(record).lower()
    score = 0.0

    if int(record["exit_code"]) != 0:
        score += 0.45
    if any(re.search(pattern, observation) for pattern in ERROR_PATTERNS):
        score += 0.35
    if ("pass" in expectation or "success" in expectation or "without" in expectation) and any(
        re.search(pattern, observation) for pattern in ERROR_PATTERNS
    ):
        score += 0.20
    if ("find" in expectation or "include" in expectation or "printed" in expectation) and not observation.strip():
        score += 0.25
    if int(record["exit_code"]) == 0 and any(re.search(pattern, observation) for pattern in SUCCESS_PATTERNS):
        score -= 0.15
    if int(record["exit_code"]) == 0 and not observation.strip() and ("without output" in expectation or "without syntax" in expectation):
        score -= 0.10

    return float(max(0.0, min(1.0, score)))


def keyword_residual(record: dict[str, Any]) -> float:
    observation = _observation(record).lower()
    if any(re.search(pattern, observation) for pattern in ERROR_PATTERNS):
        return 1.0
    return 0.0


def structured_residual(record: dict[str, Any], include_exit_code: bool = True, include_test_status: bool = True, include_error_keywords: bool = True, include_expectation: bool = True) -> float:
    observation = _observation(record).lower()
    expectation = _expectation(record).lower() if include_expectation else ""
    action = str(record.get("action") or "").lower()
    score = 0.0
    weight = 0.0

    if include_exit_code:
        weight += 0.30
        if int(record.get("exit_code") or 0) != 0:
            score += 0.30

    if include_error_keywords:
        weight += 0.25
        if any(re.search(pattern, observation) for pattern in ERROR_PATTERNS):
            score += 0.25

    if include_test_status:
        weight += 0.20
        if any(token in action for token in ["pytest", "test", "make"]) or "test" in expectation:
            if any(token in observation for token in ["failed", "failures", "error"]):
                score += 0.20
            elif any(token in observation for token in ["passed", "success", "ok"]):
                score -= 0.05

    weight += 0.15
    if any(token in action for token in ["grep", "find", "cat", "ls"]) and not observation.strip():
        score += 0.15

    if include_expectation:
        weight += 0.10
        if ("without" in expectation or "success" in expectation) and any(re.search(pattern, observation) for pattern in ERROR_PATTERNS):
            score += 0.10

    return float(max(0.0, min(1.0, score / max(weight, 1e-9))))


def calculate_scores(config_path: str, dataset_path: str | None = None, output_path: str | None = None) -> tuple[Path, str]:
    config = load_simple_yaml(config_path)
    dataset = Path(dataset_path or config["dataset"]["output_path"])
    output = Path(output_path or config["analysis"]["scores_path"])
    records = read_jsonl(dataset)

    require_model = bool(config.get("residual", {}).get("require_embedding_model", False))
    embedding_scores, embedding_backend = embedding_residuals(records, str(config["residual"]["embedding_model"]), require_model=require_model)
    tfidf_scores = tfidf_residuals(records)
    emb_weight = float(config["residual"]["hybrid_embedding_weight"])
    structured_weight = float(config["residual"].get("hybrid_structured_weight", config["residual"].get("hybrid_rule_weight", 0.40)))

    rows = []
    for record, embedding_score, tfidf_score in zip(records, embedding_scores, tfidf_scores):
        rule_score = rule_residual(record)
        keyword_score = keyword_residual(record)
        structured_score = structured_residual(record)
        hybrid_score = emb_weight * embedding_score + structured_weight * structured_score
        observation = _observation(record)
        rows.append(
            {
                "task_id": record["task_id"],
                "step_id": record["step_id"],
                "trajectory_id": record.get("trajectory_id", record["task_id"]),
                "task_category": record.get("task_category", record.get("scenario", "pilot")),
                "step_type": record.get("step_type", record.get("scenario", "pilot")),
                "trajectory_stage": record.get("trajectory_stage", "unknown"),
                "scenario": record.get("scenario", ""),
                "action": record["action"],
                "expected_outcome": _expectation(record),
                "planned_next_action_if_expected": record.get("planned_next_action_if_expected", ""),
                "actual_observation": observation,
                "exit_code": record["exit_code"],
                "reasoning_needed": _label(record),
                "annotation_confidence": record.get("annotation_confidence", ""),
                "label_reason": record["label_reason"],
                "data_source": record.get("data_source", "local_pilot"),
                "source_file": record.get("source_file", ""),
                "observation_length": len(observation),
                "command_success": int(record["exit_code"]) == 0,
                "always_reason_score": "1.000000",
                "never_reason_score": "0.000000",
                "keyword_residual": f"{keyword_score:.6f}",
                "tfidf_residual": f"{tfidf_score:.6f}",
                "embedding_residual": f"{embedding_score:.6f}" if math.isfinite(embedding_score) else "nan",
                "rule_residual": f"{rule_score:.6f}",
                "structured_residual": f"{structured_score:.6f}",
                "hybrid_no_exit_code": f"{(emb_weight * embedding_score + structured_weight * structured_residual(record, include_exit_code=False)):.6f}",
                "hybrid_no_test_status": f"{(emb_weight * embedding_score + structured_weight * structured_residual(record, include_test_status=False)):.6f}",
                "hybrid_no_error_keywords": f"{(emb_weight * embedding_score + structured_weight * structured_residual(record, include_error_keywords=False)):.6f}",
                "hybrid_no_expectation_text": f"{(emb_weight * embedding_score + structured_weight * structured_residual(record, include_expectation=False)):.6f}",
                "hybrid_residual": f"{hybrid_score:.6f}",
                "embedding_backend": embedding_backend,
            }
        )

    write_csv(output, rows)
    return output, embedding_backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phase1_residual.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    args = parser.parse_args()

    output, backend = calculate_scores(args.config, args.dataset, args.output)
    print(f"Wrote residual scores to {output} using embedding backend: {backend}")


if __name__ == "__main__":
    main()
