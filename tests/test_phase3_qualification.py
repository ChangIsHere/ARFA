import json
from pathlib import Path

from src.phase3_arfa.qualify_pair import qualify


def _summary(policy: str, success: int, calls: int, latency: float, parse_errors: int = 0) -> dict:
    fast_calls = calls if policy == "always_fast" else 0
    slow_calls = calls if policy == "always_slow" else calls
    return {
        "task_count": 20,
        "success_count": success,
        "total_model_calls": calls,
        "fast_model_calls": fast_calls,
        "slow_model_calls": slow_calls,
        "total_tokens": 1000,
        "total_model_latency_seconds": latency,
        "model_request_error_count": 0,
        "parse_error_count": parse_errors,
    }


def _write_cell(root: Path, pair: str, policy: str, summary: dict) -> None:
    cell = root / "development" / pair / policy
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = [
        "task_id,success,total_task_wall_seconds,model_calls,fast_model_calls,slow_model_calls,total_tokens",
        *[
            f"task-{index},{index < summary['success_count']},1,{summary['total_model_calls'] // 20},0,{summary['total_model_calls'] // 20},50"
            for index in range(20)
        ],
    ]
    (cell / "metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_qualification_accepts_faster_reliable_fast_model(tmp_path: Path) -> None:
    pair = "gemma_llama"
    _write_cell(tmp_path, pair, "always_slow", _summary("always_slow", 8, 40, 80.0))
    _write_cell(tmp_path, pair, "always_fast", _summary("always_fast", 7, 40, 40.0))
    _write_cell(tmp_path, pair, "residual", _summary("residual", 8, 42, 60.0))

    report = qualify(tmp_path, pair)

    assert report["eligible_for_full_development"] is True
    assert report["measurements"]["fast_latency_reduction"] == 0.5


def test_qualification_rejects_slow_or_unstable_fast_model(tmp_path: Path) -> None:
    pair = "gemma_phi"
    _write_cell(tmp_path, pair, "always_slow", _summary("always_slow", 8, 40, 80.0))
    _write_cell(tmp_path, pair, "always_fast", _summary("always_fast", 7, 40, 70.0, 3))
    _write_cell(tmp_path, pair, "residual", _summary("residual", 5, 50, 90.0))

    report = qualify(tmp_path, pair)

    assert report["eligible_for_full_development"] is False
    assert report["checks"]["fast_parse_error_rate_at_most_5pct"] is False
    assert report["checks"]["fast_latency_at_least_25pct_lower"] is False
