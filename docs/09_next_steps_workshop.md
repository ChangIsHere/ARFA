# Next Steps Toward Workshop Draft

## Current Milestone

Phase 1 has a complete external-trajectory evidence package. Phase 2 has completed the full always-reason baseline matrix: three local models across 200 released InterCode NL2Bash tasks, for 600 runs total. Phase 3 has not started.

## Immediate Research Tasks

1. Install and run the intended embedding backend:

```text
sentence-transformers/all-MiniLM-L6-v2
```

2. Review and improve Phase 1 annotation quality:

- manually inspect at least 20% of records
- resolve ambiguous records where possible
- keep ambiguous examples in the dataset
- report self-agreement or inter-annotator agreement

3. Improve expectation generation:

- expectations should be recorded before observations
- make expectations less templated
- add planned next action quality checks

4. Re-run Phase 1:

```bash
bash scripts/phase1_build_final_dataset.sh
bash scripts/phase1_run_final_analysis.sh
bash scripts/phase1_make_paper_artifacts.sh
```

5. Decide whether Phase 1 evidence passes the gate:

- false-fast rate <= 5%
- fast-path rate >= 15%
- safe-fast precision >= 95%
- hybrid beats simple baselines
- no major subgroup collapse
- no leakage source explains the result

## Completed Phase 2 Baseline

Completed:

- build the standard ReAct terminal-agent baseline
- log every action, expectation, observation, token count, latency, and task outcome
- validate all 200 environments and run the same frozen task set on three local models
- generate aggregate, paired, subgroup, diagnostic, and sensitivity analyses

The next research decision is whether and how to improve the Phase 1 residual policy enough to begin Phase 3:

- implement ARFA-Min fast-slow routing
- compare against ReAct
- report reasoning-call, token, and latency reductions

Do not implement Phase 3 until the residual deployment gate and the Phase 3 protocol are explicitly reviewed.

## Workshop Draft Tasks

Paper branch should gradually receive:

- polished abstract
- 1-page introduction
- related work notes with citations
- problem formulation
- Phase 1 method and annotation protocol
- Phase 1 result tables and figures
- Phase 2 baseline method, results, paired tests, and failure analysis
- limitations and threat-to-validity section

Do not move selected figures into the paper branch until the corresponding experiment run is accepted as the current evidence version.
