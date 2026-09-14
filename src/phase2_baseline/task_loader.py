from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.utils import read_jsonl


def load_tasks(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    tasks = read_jsonl(path)
    if limit and limit > 0:
        return tasks[:limit]
    return tasks
