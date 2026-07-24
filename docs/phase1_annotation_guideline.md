# Phase 1 Annotation Guideline

Assign labels without seeing any residual score.

Use three values:

- `true`: renewed reasoning is needed.
- `false`: renewed reasoning is not needed.
- `ambiguous`: the record should remain in the dataset but be excluded from primary binary metrics.

## `true`

Use `true` when the observation invalidates, weakens, or substantially changes assumptions required by the current plan, or when the next action is no longer safe and unambiguous.

Examples:

- unexpected traceback
- edit failed or was only partially applied
- unexpected test failure
- repository state contradicts the expectation
- expected file, symbol, or dependency is absent
- command succeeded but returned information that changes the plan
- current plan no longer specifies a safe next action

## `false`

Use `false` when the observation sufficiently matches the expectation and the existing plan already specifies a safe and unambiguous next action.

Examples:

- expected file or function was found
- planned edit was applied successfully
- expected test result was obtained
- dependency version matched the expectation
- routine command output confirms the current plan

## Important Rules

Command success alone does not mean reasoning is unnecessary.

Command failure alone does not automatically mean reasoning is needed.

The decision depends on expectation-observation discrepancy and whether the current plan remains valid.

Ambiguous records must not be silently converted into true or false.

At least 20% of records require a blind second annotation pass. The current automated self-pass is a placeholder until human reannotation is available.
