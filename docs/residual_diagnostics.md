# Residual Diagnostics: Combined Phase 1 / 1.5

## Purpose

The original external-trajectory Phase 1 is now supplementary. The main diagnostic
study uses the Phase 1.5 shadow collection: the agent commits its expectation
before the command executes but continues to reason after each observation.
This measures the signal without changing the agent's policy.

The questions are: does residual vary across model conditions; is it associated
with task failure within each model; and does it add information beyond observing
errors alone? These are diagnostics, not a requirement to prove a safe controller
before building one. Online value is the Phase 3 question.

Current positioning: **residual can be used as an auxiliary reference signal for
fast/slow reasoning-model invocation**. Here "can be used" describes a justified
experimental input, not a certified accurate decision rule. Offline associations
support proceeding to Phase 3 without asserting reliable agent control.

## Data And Score Locations

- Original external Phase 1 scores: `results/phase1_residual/final/residual_scores.csv`.
- Original online-committed shadow scores:
  `results/phase1_5_shadow/full/<model>/<split>/traces.jsonl`, in each executed step's
  `structured_residual`, `residual_checks`, `expectation_quality`, and `shadow_decision`.
- Current compact export: [step_scores.csv](../evidence/residual_diagnostics/step_scores.csv).
  It retains all **790 executed steps**, with model, task, split, step ID, and source
  trace path. Structured residual is recomputed and checked against the stored
  number; semantic and fused scores are computed offline. No steps are deduplicated
  across models in this export.
- [task_scores.csv](../evidence/residual_diagnostics/task_scores.csv) contains all
  **300 runs**, including the Qwen 7B run with no executed step. That run has empty
  residual fields, not invented zeros.
- [summary.json](../evidence/residual_diagnostics/summary.json) separates all-data
  descriptions from development (40), validation (20), and test (40) tasks per model.
- [model_comparisons.csv](../evidence/residual_diagnostics/model_comparisons.csv)
  pairs identical task IDs across models. The 751 deduplicated annotation items are
  a different unit and should not be confused with the 790 source steps.

The full expectation, signals, action, output, continuation, and outcome remain
in the linked local trace. CSV keys `(model, split, task_id, step_id)` identify the
step unambiguously; blank check fields mean the check was not applicable.

Regenerate with `bash scripts/phase1_export_diagnostics.sh`. This uses the cached
MiniLM model offline, never calls Ollama, and never changes labels or traces.
Input hashes and the cached embedding revision are in `source_manifest.json`.

## Formulas

For expectation text E and terminal observation O, the semantic score is

```text
r_sem = clip(1 - dot(normalize(embedding(E)), normalize(embedding(O))), 0, 2)
```

The encoder is `sentence-transformers/all-MiniLM-L6-v2`. The score is a distance,
not a calibrated probability; zero indicates identical embedding direction.
TF-IDF is not silently substituted by the current export.

Structured residual uses the checks implemented in
`src/phase1_5_shadow/residual.py:structured_residual`:

```text
r_struct = sum(w_j * mismatch_j for applicable checks j)
           / sum(w_j for applicable checks j)

exit-code mismatch:          w = 0.45, only if expectation is not "any"
stdout presence mismatch:   w = 0.20, only if expectation is not "any"
stderr presence mismatch:   w = 0.20, only if expectation is not "any"
textual error contradiction: w = 0.15, always in denominator
textual output missing:      w = 0.20, always in denominator
```

Text checks use the exact keyword rules in that function. Error contradiction
requires success-related expectation wording plus an observed error token. Missing
output requires output-related expectation wording and empty stdout. These are
heuristics, not semantic verification. Both stdout-presence and textual-output
checks can contribute; their weights must not be merged accidentally.

The historical analyzer's score construction is

```text
quality = (specified_signal_fraction + min(len(E.strip())/60, 1)
           + indicator(nonempty_continuation)) / 3
r_raw   = alpha * r_sem + (1 - alpha) * r_struct
r_full  = max(expectation_gate, r_raw)
r_obs   = min(0.55 * I(nonzero_exit) + 0.20 * I(nonempty_stderr)
              + 0.25 * I(error_token_in_observation), 1)
```

`expectation_gate` is 1 if the action/continuation is absent or fails the current
self-contained heuristic, continuation is `<REASON>`, or quality < 0.70; otherwise
0. `expectation_gate_only_score` exports it separately. It is not part of raw
expectation-observation mismatch. `<DONE>` is permitted by this heuristic.

The historical development-selected alpha is **0.0**. We retain it without refitting
on the test set. Thus `raw_full_residual_score` currently equals structured residual;
the export also retains semantic residual separately rather than claiming it helped.
The historical weight used subjective labels; the new failure outcomes do not.

During collection, the shadow decision used **structured residual only**: it
suggested a fast candidate below 0.35 when quality and continuation checks passed.
This decision did not execute a fast path. Offline fused scoring must not be
described as the decision rule that ran during collection.

## Model Association

Each model ran the same 100 tasks. Average residuals below first average steps
within a task, then average tasks, so longer failed trajectories do not dominate.

- Qwen 14B: 100 runs, 175 steps; mean structured residual 0.088; success 42%.
- Qwen 7B: 100 runs, 195 steps; mean structured residual 0.123 over 99 nonempty
  runs; success 22% over all 100 runs.
- Llama 8B: 100 runs, 420 steps; mean structured residual 0.211; success 21%.

The test-split task-failure AUC for task-mean structured residual is 0.613 for
Qwen 14B, 0.611 for Qwen 7B, and 0.597 for Llama 8B. Larger residual therefore ranks
some unsuccessful tasks above successful ones, but discrimination is modest.
Observation-only AUC is respectively 0.633, 0.606, and 0.667: there is no consistent
advantage for structured residual. Semantic-only AUC is 0.398, 0.434, and 0.530.
Adding the expectation gate does not uniformly improve this task-failure ranking:
the full score's test AUC is 0.672, 0.444, and 0.629 respectively. Gate design and
mismatch scoring therefore need separate online controls.

These AUCs use evaluator task failure, **not** the subjective label "reasoning
needed". Whole-trajectory means are retrospective. First-step residual AUC is
also reported to avoid treating future steps as an early prediction; on the test
split it is 0.582, 0.523, and 0.572 respectively. No new threshold was optimized.

The paired comparison export asks whether an alternative model succeeds on the
same tasks when a base model has a low/high first residual, using the existing
0.35 threshold. Group counts are shown; empty groups are omitted. These are
independent from-start runs, not outcomes of switching models within an agent
state. No confidence interval or causal claim is attached to these exploratory
subgroups; they must not be presented as a validated model-selection policy.

## What This Adds To The Paper

Residual is model-conditioned because both the committed expectation and the
observed trajectory depend on the model. The measurements motivate testing
model-specific calibration, instead of assuming one threshold transfers.
They also identify what remains to establish: incremental online value beyond
observation-only and commitment-only controls.

The useful narrative is diagnostic signal and limitations, then baseline cost,
then a controlled online experiment. The current results do not establish
irreplacability, safe skipping, savings, or causal model selection. See
[constraints](residual_constraints.md) and [Phase 3](04_phase3_arfa_dual_track.md).

## Association With Reviewed Labels

The stored Phase 1.5 analysis evaluates 313 non-ambiguous reasoning-necessity items
from 40 test tasks. Full residual AUC is 0.743 (task-clustered bootstrap 95% CI
0.681-0.811); raw residual without the expectation gate has AUC 0.739 (0.685-0.791).
This provides preliminary evidence of association with the existing reviewed
labels, separate from the evaluator-outcome association above. AUC is not an
accuracy percentage or a probability that a fast decision is safe.

The expectation-gate-only AUC is 0.619; full residual gains 0.124 (paired
task-bootstrap 95% CI 0.092-0.159). However, learned observation-only AUC is 0.749;
full-minus-learned-observation gain is -0.006 (95% CI -0.087 to 0.082). Thus this
does not establish superiority over the stronger observation baseline.

The project owner reports six-person review of AI-assisted labels. This is not a
separately stored six-reviewer gold dataset, and the earlier AI prompt defect's
impact is unmeasured. Report these associations with that qualification; do not
call them confirmed independent-human reliability. The 23+6 human pilot concerns
guideline agreement and does not establish the full packet's residual association.

[annotation_association.json](../evidence/residual_diagnostics/annotation_association.json)
publishes the relevant stored results, denominators, intervals, and source hash.
No reannotation, threshold change, or new performance test is implied by this export.
