# ARFA

**Residual-Guided Model Selection for Terminal-Based Coding Agents**

ARFA studies one decision in terminal agents: after a command finishes, should
the next reasoning turn use a fast model or a slow model? Before execution, the
agent records the expected outcome and observable signals. After execution, it
computes an expectation-observation residual and uses that score to select the
next model.

## Current Result

The evidence supports a conditional result: residual-guided routing can reduce
slow-model use and end-to-end cost, but the gain depends on the selected model
pair.

On the reserved 100-task InterCode NL2Bash split:

- **Gemma 4B/12B:** residual improved success by 9 percentage points over
  always-slow, reduced wall time by 21.5%, reduced tokens by 12.7%, and reduced
  slow-model calls by 43.5% (`p=0.0225`).
- **Qwen 7B/14B:** residual preserved always-slow success within 1 point while
  reducing wall time by 14.4%, tokens by 6.3%, and slow-model calls by 41.3%.
- **Qwen 7B/Gemma 12B:** residual improved success by 4 points over always-slow,
  reduced wall time by 33.5%, tokens by 25.8%, and slow-model calls by 49.5%.
- **Gemma 4B/Qwen 14B:** residual reduced wall time by 29.8% and tokens by 7.3%
  but lost 3 success points, showing that model size alone does not identify a
  useful fast track.

The four-pair development study also includes Llama and Phi. In both pairs,
residual reduced slow calls but increased total time or tokens because the fast
model produced more steps, repeated actions, or parse failures.

## Evidence At A Glance

- **Offline residual diagnostics:** 300 shadow trajectories, three model
  conditions, 100 tasks, and 790 executed terminal steps.
- **Standalone model characterization:** eight models on the same 200 tasks,
  producing 1,600 attempts.
- **Native-pair development study:** four pairs, three policies, and 100 shared
  tasks per cell, producing 1,200 attempts.
- **Reserved final policy study:** Qwen and Gemma, six policies, and 100 tasks
  per cell, producing 1,200 attempts with zero model-request failures.
- **Cross-family study:** two mixed pairs, five policies, and 100 tasks per cell,
  producing 1,000 attempts.
- **Handoff study:** three cross-model state-transfer modes on 100 shared tasks,
  producing 300 attempts.

The main experiments therefore contain 5,600 task-level runs, including the 300
shadow trajectories. All current task results use InterCode NL2Bash. Transfer to
a pinned second benchmark remains future work.

## Start Here

- [Manuscript handoff](docs/README_FOR_WRITING.md): paper question,
  contributions, narrative, outline, and claims.
- [Experiment results](docs/EXPERIMENT_RESULTS.md): all paper-facing numbers in
  one place, with residual-centered percentage comparisons.
- [Evidence index](evidence/README.md): exact files behind each table and claim.
- [Publication extension plan](docs/phase3_publication_experiment_plan.md):
  external benchmark, repeated energy measurement, and residual ablations.

## Repository Map

- `src/`: residual scoring, agents, routing policies, runners, and analysis.
- `config/`: frozen and development experiment configurations.
- `data/phase2/`: the 200-task InterCode manifest used by the online studies.
- `evidence/`: compact paper-facing metrics, confidence intervals, audits, and
  provenance.
- `docs/`: writing guide, methods, protocols, tables, figures, and next steps.
- `deployment/nrp/`: bounded GPU Jobs and runners used for the recorded NRP
  experiments.
- `results/`: ignored local raw traces and checkpoints. These are not required
  to read the paper-facing results.

## Reproduce And Verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Repository checks:

```bash
bash -n scripts/*.sh deployment/nrp/*.sh
git diff --check
```

Tasks come from [InterCode](https://github.com/princeton-nlp/intercode). Model
weights, caches, raw traces, credentials, kubeconfig, and API tokens are not
committed.
