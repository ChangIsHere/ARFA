from pathlib import Path

from src.phase1_residual.analyze_residual import analyze
from src.phase1_residual.dataset_builder import build_records
from src.phase1_residual.residual_calculator import calculate_scores
from src.common.utils import write_jsonl


def test_phase1_end_to_end(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    scores = tmp_path / "scores.csv"
    metrics = tmp_path / "metrics.json"
    false_fast = tmp_path / "false_fast.jsonl"
    distribution = tmp_path / "distribution.png"
    confusion = tmp_path / "confusion.png"
    report = tmp_path / "report.md"
    config = tmp_path / "phase1.yaml"

    write_jsonl(dataset, [record.to_dict() for record in build_records(50)])
    config.write_text(
        f"""
dataset:
  min_records: 50
  output_path: {dataset}
residual:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  embedding_fallback: tfidf
  embedding_threshold: 0.45
  rule_threshold: 0.50
  hybrid_threshold: 0.50
  hybrid_embedding_weight: 0.60
  hybrid_rule_weight: 0.40
analysis:
  scores_path: {scores}
  metrics_path: {metrics}
  false_fast_path: {false_fast}
  distribution_plot_path: {distribution}
  confusion_matrix_path: {confusion}
  report_path: {report}
""",
        encoding="utf-8",
    )

    calculate_scores(str(config), str(dataset), str(scores))
    payload = analyze(str(config))
    assert payload["record_count"] == 50
    assert Path(report).exists()
