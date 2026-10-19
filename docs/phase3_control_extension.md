# Phase 3 Control Extension

## Why this extension

The offline reviewed-label analysis reports ROC-AUC 0.743 for full residual and
0.749 for learned observation-only. The two intervals overlap. The present
online development runs compare residual with always-slow and always-fast, but
not with an equally capable observation-only router. They therefore do not yet
isolate the value of expectation-observation comparison.

## First experiment: shared InterCode tasks

Start with Qwen 7B/14B on the same 100 development task IDs used in Phase 3.
Keep the prompt, evaluator, Docker images, context length, temperature, step
cap, trajectory concurrency, GPU class, and model artifacts fixed across
policies. Use a new result root; do not merge these runs with the earlier
Qwen development records.

The comparison set is `residual`, `always_slow`, `always_fast`,
`learned_observation_only`, `observation_only`, `expectation_gate_only`,
`random_budget`, and `fixed_budget`. The learned control reads only the
terminal observation and exit code. It fits on 287 binary-labeled Phase 1.5
shadow-development items, then chooses its threshold on 150 shadow-validation
items. Its calibration record is
[`learned_observation_validation.json`](../evidence/phase3_design/learned_observation_validation.json).
Those labels are AI-assisted with owner-reported human review; separate raw
reviewer files are not available. The new online development run is a design
comparison, not independent held-out confirmation: some training and online
development items share task IDs, although they are different trajectories.

`random_budget` and `fixed_budget` use the configured fast-call budget, not
the residual score or terminal output. Their actual fast-call fractions must
be reported. A target budget is not evidence of a matched achieved budget;
for any budget-matched claim, calibrate these controls on development and
verify their observed fast-call rates against residual before freezing.

Primary readouts are task success, paired task-wall time, total tokens,
model calls by role, retries, actual fast-call rate, and GPU energy estimated
from sampled power. Report denominators and paired bootstrap intervals.
Wall time and energy are different outcomes; do not infer one from the other.

Run one bounded NRP Job per policy. The Job has a deadline and the runner
deletes it on exit. Check both model residents before and after every run.
The output directory is `results/phase3_arfa/nrp/nrp-qwen-controls-v1/`.
First run a 12-task smoke, then the complete 100-task development cell:

```bash
bash deployment/nrp/run_qwen_controls.sh residual smoke
bash deployment/nrp/run_qwen_controls.sh residual full
bash deployment/nrp/run_qwen_controls.sh learned_observation_only full
```

Repeat `full` for the other six policies. `full` performs its own smoke first
and resumes only from matching provenance. The experiment is incomplete until
all intended cells contain 100 unique matching task IDs.
Run the analysis with `--strict-controls` to reject a mismatch in task IDs,
config, code, model identities, or Docker images. Its `control_audit` reports
achieved fast-call rates and whether the budget controls are actually within
five percentage points of residual:

```bash
.venv/bin/python -m src.phase3_arfa.analyze_results \
  --result-root results/phase3_arfa/nrp/nrp-qwen-controls-v1 \
  --split development --pair qwen --strict-controls
```

## Beyond the first pair

If Qwen controls run cleanly, apply the same protocol to Gemma 4B/12B and
retain Llama/Phi as pair-dependence cases. Select policy parameters only on
development. Freeze code, task IDs, model hashes, thresholds, seeds, budget,
concurrency, Docker images, and analysis before using the reserved 100-task
InterCode final split. Evaluate that split once; a disappointing result is
still a result.

For external validity, use a second terminal/coding environment rather than
simply increasing InterCode task counts. [Terminal-Bench 2.0](https://github.com/harbor-framework/terminal-bench)
is a close domain match but has 89 tasks, so the earlier plan's "at least 100"
target does not apply to that complete benchmark. Verify tasks with the
official Harbor oracle and keep its evaluator separate from InterCode.
[SWE-bench](https://github.com/swe-bench/SWE-bench) is a complementary
bug-fixing benchmark with a heavier environment. Neither benchmark has yet
been run in this extension.
