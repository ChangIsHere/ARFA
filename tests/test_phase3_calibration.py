from pathlib import Path

from src.common.utils import write_csv
from src.phase3_arfa.calibrate_threshold import calibrate


def _rows(successes: int, wall: float, slow_calls: int, tokens: int) -> list[dict[str, object]]:
    return [
        {
            "task_id": f"task-{index}",
            "success": index < successes,
            "total_task_wall_seconds": wall,
            "model_calls": slow_calls,
            "fast_model_calls": 0,
            "slow_model_calls": slow_calls,
            "total_tokens": tokens,
        }
        for index in range(4)
    ]


def test_calibration_selects_only_quality_preserving_faster_candidate(tmp_path: Path) -> None:
    root = tmp_path / "results"
    pair_root = root / "development" / "llama"
    write_csv(pair_root / "always_slow" / "metrics.csv", _rows(2, 10.0, 4, 100))
    write_csv(pair_root / "residual" / "metrics.csv", _rows(3, 12.0, 3, 110))
    candidate = root / "calibration" / "threshold_0_20" / "development" / "llama" / "residual"
    write_csv(candidate / "metrics.csv", _rows(2, 8.0, 3, 90))

    report = calibrate(root, "llama", "development", [0.20, 0.35], 4, 100, 7)

    assert report["selection_succeeded"] is True
    assert report["selected_threshold"] == 0.20
    rejected = next(row for row in report["candidates"] if row["threshold"] == 0.35)
    assert rejected["eligible"] is False
    assert "wall_not_reduced" in rejected["reason"]


def test_calibration_reports_no_selection_when_quality_drops(tmp_path: Path) -> None:
    root = tmp_path / "results"
    pair_root = root / "development" / "llama"
    write_csv(pair_root / "always_slow" / "metrics.csv", _rows(3, 10.0, 4, 100))
    candidate = root / "calibration" / "threshold_0_20" / "development" / "llama" / "residual"
    write_csv(candidate / "metrics.csv", _rows(2, 8.0, 3, 90))

    report = calibrate(root, "llama", "development", [0.20], 4, 100, 7)

    assert report["selection_succeeded"] is False
    assert report["selected_threshold"] is None
    assert "success_below_slow" in report["candidates"][0]["reason"]
