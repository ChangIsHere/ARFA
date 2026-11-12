# Reproducibility

This guide describes the repository state used for the reported ARFA results.
The compact artifacts under `evidence/` are tracked in Git. Raw conversations,
model weights, caches, and checkpoints are intentionally excluded because of
their size and, in the case of cluster credentials, security requirements.

## Software Setup

The checked local analysis environment uses Python 3.10.11. The top-level Python
packages are pinned in `requirements.txt`; transitive inference dependencies are
also recorded in the runtime provenance files for each GPU study. Local agent
runs require Docker and Ollama. NRP runs use Kubernetes Jobs through `kubectl`;
no kubeconfig or authentication material is stored in this repository.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

The shell and configuration files can be checked without starting an
experiment:

```bash
bash -n scripts/*.sh deployment/nrp/*.sh
git diff --check
```

## Task Data

The online studies use 200 tasks from the released InterCode NL2Bash suite:

- `data/phase2/intercode_nl2bash_official_200.jsonl`
- `data/phase2/intercode_nl2bash_official_200_summary.json`

Each record preserves the source repository, source commit, filesystem version,
source index, task instruction, and evaluator definition. The 100-task
development and reserved splits are fixed in `data/phase1_5/task_splits.json`.
The repository does not describe either subset as an upstream official test
split.

To rebuild the task manifest from a local InterCode checkout:

```bash
scripts/phase2_build_official_dataset.sh
scripts/phase2_validate_official_environment.sh
```

## Experiment Stages

### Residual diagnostics

The shadow pipeline records a pre-execution commitment, executes the command,
and computes residual scores without changing the agent's decisions.

```bash
scripts/phase1_5_run_shadow_matrix.sh
scripts/phase1_export_diagnostics.sh
```

Tracked outputs are under `evidence/residual_diagnostics/`. The historical pilot
and annotation audit are retained under `evidence/phase1_5_pilot/`.

### Standalone model characterization

Phase 2 runs eight single-model ReAct agents on the same 200 tasks. These runs
characterize the candidate models; they are not the matched causal controls for
the routing policy.

```bash
scripts/phase2_run_paper_matrix.sh
scripts/phase2_make_paper_artifacts.sh
scripts/phase2_build_evidence_bundle.sh
```

The compact package is stored in `evidence/phase2_baselines/`.

### Dual-model routing

Phase 3 compares residual routing with matched always-slow, always-fast,
learned-observation, expectation-gate, and random-budget policies. The local
entry point is:

```bash
scripts/phase3_run_arfa.sh --config config/phase3_arfa.yaml
scripts/phase3_analyze.sh
```

The recorded GPU studies used the bounded Job runners in `deployment/nrp/`.
Those runners create an experiment Job, write checkpoints and traces to the
configured persistent volume, collect provenance, and terminate the workload
after completion. They should be reviewed with the namespace owner before use
on another cluster.

## Evidence Layout

- `evidence/phase3_design/native_pairs/`: four native development pairs.
- `evidence/phase3_final/`: frozen Qwen and Gemma policy matrices.
- `evidence/phase3_mixed_pairs/`: two cross-family policy matrices.
- `evidence/phase3_handoff/`: three handoff modes.
- `evidence/phase3_design/supplementary/`: routing controls and cost-reversal
  analysis.

Each final evidence directory contains a combination of policy summaries,
task-level metrics, paired analyses, audits, environment records, and SHA-256
manifests. The task-level CSV files omit full model conversations but retain the
outcomes and cost fields needed for paired reanalysis. The manifests identify
the code, configuration, model metadata, and source artifacts used for the
corresponding result.

Freeze manifests describe the source state recorded by the experiment. They
should not automatically be checked against current `main`. In particular, the
mixed-pair matrix records commit `7184111` for the agent source and separately
hashes the exact mixed-pair configurations and runners published with its
evidence package. The later handoff study records the source state that
introduced its three transfer modes. Artifact manifests hash the compact result
files themselves and can be checked directly from the current repository.

## Result Interpretation

Compare latency only within a matched matrix. Several development matrices ran
on different GPU types, so their absolute seconds are not pooled. Success means
acceptance by the task evaluator, not exact command matching. Token counts are
reported by the inference runtime. Existing GPU samples document the runtime
environment but are not treated as policy-level causal energy measurements.

The exact reported values and their source directories are listed in
[`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md).
