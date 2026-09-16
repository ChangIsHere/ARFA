# Next Steps Toward Workshop Draft

## Current Milestone

Phase 1 has a complete external-trajectory evidence package. Phase 2 has completed the full always-reason baseline matrix: three local models across 200 released InterCode NL2Bash tasks, for 600 runs total. Phase 1.5 completed 300 shadow runs and 751 human-reviewed, AI-assisted blind labels. Its engineering-readiness gate passed, so Phase 3-P implementation can proceed on development tasks.

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

The next implementation milestone is the guarded Phase 3-P pilot:

- implement fused ARFA fast-slow routing
- limit the pilot to development tasks and one consecutive fast step
- fail open to slow reasoning after errors or uncertainty
- compare against ReAct and observation-only routing
- report task success, recovery, calls, tokens, and latency

Do not consume the reserved `phase3_final` split during Phase 3-P. Formal Phase 3 starts only after the exploratory controller and its protocol are reviewed.

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
