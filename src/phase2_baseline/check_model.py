from __future__ import annotations

import argparse

from src.common.config import load_simple_yaml
from src.phase2_baseline.model_client import OpenAICompatibleClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the Phase 2 local model endpoint.")
    parser.add_argument("--config", default="config/phase2_baseline.yaml")
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    client = OpenAICompatibleClient.from_config(config["baseline"])
    result = client.chat(
        [
            {"role": "system", "content": "Reply with JSON only."},
            {"role": "user", "content": "{\"status\":\"ok\"}"},
        ]
    )
    print(result.content)


if __name__ == "__main__":
    main()
