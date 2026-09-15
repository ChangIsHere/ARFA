# Phase 1.5 Shadow Residual Protocol

## Purpose

Phase 1.5 repairs the two main limitations of Phase 1: retrospective template expectations and heuristic labels coupled to structured residual features. It does not repeat the original 10,500-step offline experiment.

## Experimental Boundary

The agent generates `current_plan`, `action`, `expected_outcome`, structured `expected_signals`, and `next_action_if_expected` before command execution. Ollama native JSON-schema decoding enforces valid output and the allowed signal enums. A continuation must be an independently executable shell command, `<DONE>`, or `<REASON>` when the next command depends on unseen observation content. The residual controller then records a hypothetical decision. It never controls execution; the large model still reasons after every observation.

Current actions must also be self-contained. To prevent repeated failure loops from duplicating annotation content, the collector stops a task before executing the third occurrence of an identical command.

An action that depends on missing stdin is handled by a shared safety gate and can never become a fast candidate. This rule must also be applied to the no-expectation baseline so its contribution is not attributed to residual scoring.

Phase 1.5 must never execute tasks assigned to `phase3_final`.

Protocol version `phase1.5-v1.1` is in pilot review. `protocol.locked` remains false until the pilot labels and independent agreement check are complete. Formal shadow collection is forbidden before a tracked freeze manifest records the config, prompt, schema, splits, annotation guide, analyzer, runner, and residual implementation hashes.

## Frozen Splits

- `shadow_development`: 40 tasks for representation development.
- `shadow_validation`: 20 tasks for threshold selection.
- `shadow_test`: 40 tasks for one held-out Phase 1.5 evaluation.
- `phase3_final`: 100 tasks excluded from all Phase 1.5 fitting and threshold selection, reserved for end-to-end routing. Phase 2 already provides always-reason baseline runs on the full 200-task suite.

All three model traces for one task remain in the same split.

## Leakage Controls

The blind annotation packet excludes model identity, split, residual scores, shadow decisions, future actions, final reward, final answer, and task success. Labels must not be inferred from the released gold command.

The annotator sees only the task, pre-execution commitment, action, actual observation, and shell exit code.

## Acceptance Gate

Phase 3 remains locked unless the held-out shadow test satisfies all of the following:

- safe-fast precision at least 95%
- fast-path rate at least 15%
- false-fast rate at most 5%
- full expectation-conditioned residual beats the no-expectation baseline
- full residual gains at least 0.02 ROC-AUC over no-expectation and its task-bootstrap 95% interval excludes zero
- at least 100 binary-labeled held-out items
- expectation-mismatch ROC-AUC at least 0.80
- Cohen's kappa at least 0.70 on both labels over an independently annotated 25% sample
- no eligible model or task-category subgroup below 85% safe-fast precision or above 10% false-fast rate

Only subgroups with at least 20 held-out items enter the subgroup gate. All smaller groups are reported but treated as descriptive.

Threshold constraints are frozen before collection. Validation threshold selection must satisfy both the false-fast and safe-fast constraints before maximizing fast-path coverage. If no threshold is feasible, the analyzer records infeasibility, falls back to always-reason for descriptive test evaluation, and fails the Phase 3 gate. Failure to pass must not be repaired by retuning on the held-out test.

Blank labels are incomplete. `ambiguous` is a completed annotation that is excluded from binary metrics and reported separately. Inter-annotator analysis uses three-class Cohen's kappa and also reports non-ambiguous binary-pair coverage.

Task-clustered bootstrap 95% confidence intervals accompany safe-fast precision, fast-path rate, false-fast rate, ROC-AUC, AUC gain, and eligible subgroup results. Acceptance remains based on the preregistered point estimates, with denominators and intervals reported alongside them.

## Reproducible Commands

```bash
bash scripts/phase1_5_build_splits.sh
bash scripts/phase1_5_run_pilot.sh
bash scripts/phase1_5_audit_collection.sh
bash scripts/phase1_5_build_annotation_packet.sh --results-root results/phase1_5_shadow/pilot --output-dir results/phase1_5_shadow/pilot_annotation
# Complete both pilot annotation packets, then run the pilot analysis.
bash scripts/phase1_5_analyze_annotations.sh \
  --annotations results/phase1_5_shadow/pilot_annotation/blind_annotation_packet.jsonl \
  --secondary-annotations results/phase1_5_shadow/pilot_annotation/blind_annotation_packet_secondary_25pct.jsonl \
  --source-map results/phase1_5_shadow/pilot_annotation/private_source_map.jsonl \
  --output-dir results/phase1_5_shadow/pilot_analysis
bash scripts/phase1_5_freeze_protocol.sh
bash scripts/phase1_5_run_shadow_matrix.sh
# Complete both formal annotation packets, then run held-out analysis once.
bash scripts/phase1_5_analyze_annotations.sh
bash scripts/phase1_5_build_evidence_bundle.sh
```
