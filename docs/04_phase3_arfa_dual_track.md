# Phase 3 Online Dual-Model Experiment

Phase 3 tests whether execution residual can choose between a fast and slow model
while retaining task success and reducing cost. The controller is implemented and
uses the same 200 InterCode NL2Bash tasks and Docker evaluator as Phase 2.

This is the project's primary routing experiment. Phase 2 measures models used
alone; Phase 3 isolates the effect of the routing policy with same-environment
matched controls.

## Current Status

All four development pairs are complete. Each pair used 100 shared tasks under
residual, always-slow, and always-fast policies, for 1,200 policy-task attempts.

- Qwen residual matched always-slow at 41% success and reduced slow calls by
  40.86%. Its 4.87% wall-time reduction was inconclusive.
- Llama residual reached 19% success versus 16% for always-slow, but increased
  wall time by 13.78% despite using 22.64% fewer slow calls.
- Gemma's development-selected threshold reached 29% versus 26% and reduced
  slow calls, wall time, and tokens by about 48.1%, 30.6%, and 30.6%.
- No Phi threshold passed the saved development rule. Its best-success setting
  reached 33% versus 30%, reduced slow calls by 28.43%, and increased wall time
  by 9.83%.

See the [four-pair evidence index](../evidence/phase3_design/README.md). These are
development results; the reserved final split remains unused.

## Model Pairs

| Pair | Fast model | Slow model |
| --- | --- | --- |
| Qwen | Qwen2.5-Coder 7B | Qwen2.5-Coder 14B |
| Llama | Llama 3.2 3B | Llama 3.1 8B |
| Gemma | Gemma 3 4B | Gemma 3 12B |
| Phi | Phi-4 Mini 3.8B | Phi-4 14B |

Every pair uses the same prompt, 8K context, temperature 0, 12-step cap, task
manifest, and evaluator. Phase 2 contains the corresponding standalone model
characterization; the matched Phase 3 policies provide the main comparison.

## Online Controller

The slow model handles the first turn. Each model response commits to an action,
expected outcome, expected exit code, stdout/stderr presence, and possible next
action before execution. After the terminal command runs, ARFA computes the
structured expectation-observation residual:

```text
R = sum_j w_j * mismatch_j / sum_j w_j
```

The checks cover exit-code mismatch, stdout/stderr-presence mismatch, an error
contradicting an expected success, and missing output when visible output was
expected. With a development threshold `tau`:

```text
R < tau  -> next reasoning call uses the fast model
R >= tau -> next reasoning call uses the slow model
```

Qwen and Llama used `tau=0.35`. Gemma selected `tau=0.30` on development tasks.
Phi's reported `tau=0.15` result is diagnostic because no tested threshold met
the saved selection rule. Missing signals, malformed model output, or low
expectation quality route to the slow model. At most one consecutive fast call
is allowed in the current protocol.
Every step records the residual, mismatch checks, selected model, tokens, model
latency, command latency, router overhead, and next routing decision.

## Data Split

- `development`: the 100 former shadow tasks; threshold and policy development
  may use these tasks.
- `final`: 100 reserved end-to-end tasks; the runner refuses access while
  `use_phase3_final_tasks` is false.

The final split is run only after the controller, threshold, prompt, comparison
policies, and analysis are frozen. It cannot be run with a task limit or threshold
override.

## Policies

- `residual`: the main ARFA policy.
- `always_slow`: matched slow-model control under the Phase 3 prompt.
- `always_fast`: matched fast-model control.
- `observation_only`: fast/slow routing without expectation text.
- `fixed_switch`: alternating slow and fast calls without residual.

These controls separate the benefit of residual from model size, Phase 3 prompt
overhead, and generic switching. Historical Phase 2 runs explain why each pair was
chosen and show standalone capability. Matched Phase 3 controls are the primary
baselines for routing-effect, latency, and cost claims.

## Running Locally

```bash
bash scripts/phase3_runtime_preflight.sh
bash scripts/phase3_run_development.sh qwen residual
```

Both commands run from the repository root. The experiment checkpoints after
every task and `--resume` is enabled by the wrapper. A one-task smoke run can be
started with:

```bash
bash scripts/phase3_run_development.sh qwen residual --task-limit 1
```

Smoke outputs go to a separate directory and cannot contaminate the complete
development run. To run the three core Qwen policies after the smoke check:

```bash
bash scripts/phase3_run_development_matrix.sh qwen core
```

To collect the residual policy for all four pairs, use `all residual`. The full
four-pair, five-policy matrix is `all full`; it is expensive and runs serially.
The completed GPU development runs used the bounded jobs documented under
`deployment/nrp/`; local and GPU timing results must remain separate. The Qwen
success margin and cost-per-success analysis remain specified in
[the pre-run study protocol](phase3_study_protocol.md).

For an overnight Mac run that keeps the machine awake, performs preflight first,
and saves a visible terminal log, use:

```bash
bash scripts/phase3_run_overnight.sh all residual
```

Re-running the same command resumes completed task cells instead of starting them
again. Closing the terminal still stops the active process, but saved tasks remain.

After two or more policies finish, generate paired summaries with:

```bash
bash scripts/phase3_analyze.sh qwen development
```
