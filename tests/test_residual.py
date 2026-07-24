from src.phase1_residual.dataset_builder import build_records
from src.phase1_residual.residual_calculator import rule_residual


def test_dataset_has_minimum_records():
    assert len(build_records(50)) == 50


def test_rule_residual_detects_failure():
    record = build_records(2)[1].to_dict()
    assert rule_residual(record) >= 0.5


def test_rule_residual_allows_success():
    record = build_records(1)[0].to_dict()
    assert rule_residual(record) < 0.5
