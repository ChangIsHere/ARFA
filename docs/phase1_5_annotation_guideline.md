# Phase 1.5 Blind Annotation Guideline

Annotate each item without residual scores, model identity, final reward, future actions, or task outcome.

## Expectation Match

- `match`: the observation is compatible with the concrete pre-execution expectation and expected signals.
- `mismatch`: the observation contradicts a material part of the expectation.
- `ambiguous`: the expectation or observation is insufficient to decide.

## Slow Reasoning Needed

- `no`: the committed self-contained `next_action_if_expected` (or `<DONE>`) remains safe and appropriate after the observation.
- `yes`: the observation invalidates the commitment, exposes an error, changes a required assumption, or makes the next action unsafe.
- `ambiguous`: the packet does not contain enough evidence to judge safe continuation.

`<REASON>` is an explicit declaration that the model could not safely precommit a continuation. Label slow reasoning as needed unless the task was already complete and `<DONE>` should clearly have been used.

Do not label `yes` merely because a command exits nonzero. A predicted nonzero result can match the expectation and preserve the plan. Do not label `no` merely because exit code is zero.

Use `annotation_confidence` values `high`, `medium`, or `low`, and explain every `ambiguous` label. A second annotator must independently label the frozen 25% packet. Do not overwrite either independent packet before the analyzer records agreement; store any later adjudication separately.

Use distinct annotator IDs. The second annotator must not inspect the primary labels. If the same person performs both pilot passes, use a different ID, a different shuffle seed, and a meaningful time gap; this is only a guideline check and cannot be reported as formal inter-annotator agreement. Formal agreement requires an independently labeled 25% subset from the complete formal packet.
