import json

import pytest

from src.phase2_baseline.build_official_dataset import build_records
from src.phase2_baseline.intercode_docker_environment import InterCodeDockerEnvironment, _decode_stream


def test_build_official_records(tmp_path):
    for fs_version, count in ((1, 2), (2, 1), (3, 1), (4, 1)):
        rows = [{"query": f"Count sample {index}", "gold": "true"} for index in range(count)]
        (tmp_path / f"nl2bash_fs_{fs_version}.json").write_text(json.dumps(rows), encoding="utf-8")

    records = build_records(tmp_path)

    assert len(records) == 5
    assert records[0]["task_id"] == "intercode-nl2bash-fs1-000"
    assert records[-1]["filesystem_version"] == 4
    assert all(record["source_commit"] for record in records)
    assert all(record["source_partition"] == "full_intercode_nl2bash_suite" for record in records)


def test_intercode_text_similarity():
    assert InterCodeDockerEnvironment._text_similarity("same output", "same output") == pytest.approx(1.0)
    assert InterCodeDockerEnvironment._text_similarity("", "") == 1.0
    assert InterCodeDockerEnvironment._text_similarity("alpha", "beta") == 0.0


def test_intercode_binary_output_uses_replacement_decoding():
    assert _decode_stream(b"gzip:\x8bpayload") == "gzip:\ufffdpayload"
