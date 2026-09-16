from __future__ import annotations

import argparse
import json

from src.common.config import load_simple_yaml


def phase3_status(config: dict) -> dict:
    if config.get("locked", True):
        return {
            "ready": False,
            "stage": config.get("stage", "phase3_formal"),
            "reason": config.get("reason", "Phase 3 is locked."),
        }
    return {
        "ready": True,
        "stage": config.get("stage", "phase3_p_exploratory"),
        "scope": config.get("scope", "engineering_pilot"),
        "formal_claims_enabled": bool(config.get("formal_claims_enabled", False)),
        "use_phase3_final_tasks": bool(config.get("use_phase3_final_tasks", False)),
        "router": config.get("router", {}),
        "status": "scaffold_unlocked",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or inspect the ARFA Phase 3 stage.")
    parser.add_argument("--config", default="config/phase3_arfa.yaml")
    args = parser.parse_args()
    status = phase3_status(load_simple_yaml(args.config))
    print(json.dumps(status, indent=2))
    if not status["ready"]:
        raise SystemExit(status["reason"])


if __name__ == "__main__":
    main()
