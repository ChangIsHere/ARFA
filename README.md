# ARFA

**Residual-Guided Reasoning Control for Terminal-Based Coding Agents**

ARFA studies whether the difference between an expected execution outcome and an
observed terminal result can help decide when an agent should reason again.
**Execution residuals can be used as an auxiliary decision signal for fast/slow
reasoning-model invocation.** The offline evidence motivates testing this use in
Phase 3; it does not establish reliable routing or guarantee that a plan is safe.

## Current Work

1. **Residual diagnostics (Phase 1, incorporating Phase 1.5):** 300 local shadow
   runs, three models on the same 100 tasks, 790 executed steps. Expectations were
   committed before execution. Per-step residuals and model comparisons are now
   exported together. The old external-trajectory Phase 1 study is supplementary.
2. **Always-reason baseline (Phase 2):** complete, 600 runs across three models and
   200 InterCode NL2Bash tasks. Its code, results, and writing bundle are unchanged.
3. **Online control (Phase 3):** engineering development is permitted using the
   existing lighter readiness gate. The agent/router files are still placeholders;
   no online ARFA savings or safe-routing result has been obtained.

The historical strict offline acceptance gate is **retired from the current
development workflow**, not rewritten as a pass. Frozen protocols and original
outputs are retained for reproduction. See [constraints](docs/residual_constraints.md).

## Start Here

- [Residual study, formulas, and model association](docs/residual_diagnostics.md)
- [Paper-writing handoff and supported claims](docs/README_FOR_WRITING.md)
- [Per-step residual values](evidence/residual_diagnostics/step_scores.csv)
- [Task-level residuals and outcomes](evidence/residual_diagnostics/task_scores.csv)
- [Model summaries](evidence/residual_diagnostics/summary.json)
- [Paired model comparisons](evidence/residual_diagnostics/model_comparisons.csv)
- [Phase 2 writing bundle](docs/phase2_writing_bundle/README_FOR_WRITING.md)
- [Phase 3 implementation plan](docs/04_phase3_arfa_dual_track.md)

## What The Evidence Says

In the shadow collection, Qwen 14B has a lower task-averaged structured residual
(0.088) than Qwen 7B (0.123) and Llama 8B (0.211). Their shadow task success rates
are 42%, 22%, and 21%, respectively. These are descriptive results under one
protocol, not evidence that residual determines the best model for each task.

Within the held-out shadow task split, structured-residual task-failure AUC is
approximately 0.60-0.61. Observation-only scoring is competitive. Semantic
similarity alone does not consistently identify failures, and the historical
development-selected semantic weight was zero. The next experiment must test
whether residual-informed control adds value over observation-only control.

Against the existing human-reviewed, AI-assisted reasoning-necessity labels,
full residual has held-out AUC 0.743 (task-bootstrap 95% CI 0.681-0.811; 313
binary items across 40 tasks). This is preliminary label association, subject to
the provenance and prompt limitations below, not independent human-gold validation.
The [annotation association snapshot](evidence/residual_diagnostics/annotation_association.json)
includes raw residual and competing baselines, not only the full system score.

Phase 2 success rates are 40.0% (Qwen 14B), 26.0% (Qwen 7B), and 25.5% (Llama 8B)
on its **200-task** matrix. Do not mix these with the 100-task shadow results.

## Reproduce And Inspect

Use the existing `.venv` and cached MiniLM model. The following export is offline;
it does not call an agent, relabel items, tune a threshold, or rerun Docker tasks:

```bash
bash scripts/phase1_export_diagnostics.sh
.venv/bin/python -m pytest -q
bash scripts/phase3_run_arfa.sh
```

The last command reports development readiness only; it does **not** run an ARFA
agent. The `phase3_run_exploratory.sh` compatibility entry point uses the same
configuration, `config/phase3_arfa.yaml`.

## Data And Layout

InterCode tasks and external trajectories originate from the official
[Princeton NLP InterCode repository](https://github.com/princeton-nlp/intercode).
Downloaded external files remain locally in `data/phase1/external_intercode/`.
Shadow trajectories were produced by our local agents; they are not downloaded
model answers. Full raw traces, model caches, and environments are kept locally
and are intentionally excluded from git.

```text
src/phase1_residual/             legacy external-trajectory implementation
src/phase1_5_shadow/             shadow collection and unified diagnostic export
src/phase2_baseline/             completed always-reason baseline
src/phase3_arfa/                 online-controller scaffold
config/                         runtime configs and preserved frozen protocols
scripts/                        command entry points
tests/                          focused regression tests
evidence/residual_diagnostics/  current compact numeric evidence
docs/phase2_writing_bundle/     unchanged baseline writing package
docs/archive/                   historical planning and supplementary Phase 1
data/phase1_5/annotation/        local packets, labels, and source mapping
results/phase1_5_shadow/full/    local original shadow traces
```

Collection filenames retain `phase1_5` to keep provenance paths and frozen hashes
valid. The merge is a research/documentation consolidation, not a raw-data rewrite.

Annotation provenance is **owner-reported human-reviewed AI-assisted**: six people
reportedly divided the review. Raw per-reviewer files are unavailable; stored
AI-judge agreement is not human inter-annotator agreement. A short-code prompt
defect was found in the AI annotator and corrected for future use; existing labels
were not silently regenerated. Label-dependent findings remain provisional.

`main` contains code and traceable evidence; `paper` is intended for manuscript
sections. Manuscript results should reference a code commit and artifact manifest.
See the [documentation index](docs/README.md) for current versus historical material.
