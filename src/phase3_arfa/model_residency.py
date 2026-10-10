"""Load a Phase 3 model pair and verify that both models remain resident."""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

from src.common.config import load_simple_yaml


def _api_root(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")


def _request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def inspect_pair_residency(config_path: Path, pair_name: str) -> dict[str, Any]:
    config = load_simple_yaml(config_path)
    pair = config["pairs"][pair_name]
    base_url = str(config["model"]["base_url"])
    api_root = _api_root(base_url)
    keep_alive = config["model"].get("keep_alive", -1)
    context_length = int(config["model"]["context_length"])
    required = [str(pair["fast_model"]), str(pair["slow_model"])]

    for model_name in required:
        _request_json(
            f"{api_root}/api/generate",
            {
                "model": model_name,
                "prompt": "",
                "stream": False,
                "keep_alive": keep_alive,
                "options": {"num_ctx": context_length},
            },
        )

    response = _request_json(f"{api_root}/api/ps")
    resident = sorted(str(row.get("name", "")) for row in response.get("models", []))
    missing = sorted(set(required) - set(resident))
    return {
        "phase": "phase3_model_residency",
        "pair": pair_name,
        "base_url": base_url,
        "context_length": context_length,
        "keep_alive": keep_alive,
        "required_models": required,
        "resident_models": resident,
        "missing_models": missing,
        "passed": not missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/phase3_arfa.yaml"))
    parser.add_argument("--pair", choices=("qwen", "llama", "gemma", "phi"), default="qwen")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = inspect_pair_residency(args.config, args.pair)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(
            "Both models must remain resident before a matched Phase 3 efficiency run. "
            "Use a larger-memory runtime or a smaller predeclared model pair."
        )


if __name__ == "__main__":
    main()
