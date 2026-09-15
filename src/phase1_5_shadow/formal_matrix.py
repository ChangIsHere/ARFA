from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORMAL_SPLITS = ("shadow_development", "shadow_validation", "shadow_test")


@dataclass(frozen=True)
class FormalModel:
    name: str
    slug: str


def formal_models(config: dict[str, Any]) -> list[FormalModel]:
    formal = config.get("formal", {})
    entries = [str(value) for key, value in sorted(formal.items()) if key.startswith("model_")]
    models: list[FormalModel] = []
    for entry in entries:
        name, separator, slug = entry.partition("|")
        if not separator or not name.strip() or not slug.strip():
            raise ValueError(f"Invalid formal model entry: {entry!r}")
        models.append(FormalModel(name=name.strip(), slug=slug.strip()))
    if not models:
        raise ValueError("No formal model entries are configured")
    if len({model.name for model in models}) != len(models):
        raise ValueError("Formal model names must be unique")
    if len({model.slug for model in models}) != len(models):
        raise ValueError("Formal model slugs must be unique")
    return models


def expected_trace_paths(config: dict[str, Any], result_root: str | Path) -> dict[tuple[str, str], Path]:
    root = Path(result_root)
    return {
        (model.name, split): root / model.slug / split / "traces.jsonl"
        for model in formal_models(config)
        for split in FORMAL_SPLITS
    }
