import copy
import hashlib
from pathlib import Path

import pytest

from src.common.config import load_simple_yaml
from src.phase3_arfa.freeze_final import frozen_files, verify_manifest


def _manifest(pair: str) -> dict:
    config = load_simple_yaml(f"config/phase3_final_{pair}.yaml")
    return {
        "phase": "phase3_final",
        "pair": pair,
        "protocol_version": config["protocol_version"],
        "locked": True,
        "source_commit": "test",
        "files": {
            path: {
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "bytes": Path(path).stat().st_size,
            }
            for path in frozen_files(pair)
        },
    }


@pytest.mark.parametrize("pair", ["qwen", "gemma"])
def test_final_manifest_verifies_complete_file_set(pair: str) -> None:
    verify_manifest(pair, _manifest(pair))


def test_final_manifest_rejects_missing_file() -> None:
    manifest = _manifest("qwen")
    manifest["files"].pop(next(iter(manifest["files"])))
    with pytest.raises(ValueError, match="file set"):
        verify_manifest("qwen", manifest)


def test_final_manifest_rejects_modified_hash() -> None:
    manifest = copy.deepcopy(_manifest("gemma"))
    manifest["files"][next(iter(manifest["files"]))]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed or is missing"):
        verify_manifest("gemma", manifest)
