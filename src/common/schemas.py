from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class TerminalStepRecord:
    task_id: str
    step_id: int
    instruction: str
    action: str
    expectation: str
    observation: str
    exit_code: int
    reasoning_needed: bool
    label_reason: str
    scenario: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase1FinalRecord:
    task_id: str
    step_id: int
    task_category: str
    step_type: str
    task_description: str
    state_summary: str
    current_plan: str
    action: str
    expected_outcome: str
    planned_next_action_if_expected: str
    actual_observation: str
    exit_code: int
    reasoning_needed: str
    annotation_confidence: str
    label_reason: str
    data_source: str
    trajectory_id: str
    source_file: str
    external_reward: float | None
    external_valid_action: bool | None
    trajectory_stage: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
