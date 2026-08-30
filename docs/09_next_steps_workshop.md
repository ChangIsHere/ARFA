# Next Steps Toward Workshop Draft

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

## If Phase 1 Passes

Begin Phase 2:

- build the standard ReAct terminal-agent baseline
- log every action, expectation, observation, token count, latency, and task outcome
- keep the same task definitions and evaluation protocol where possible

Then begin Phase 3:

- implement ARFA-Min fast-slow routing
- compare against ReAct
- report reasoning-call, token, and latency reductions

## Workshop Draft Tasks

Paper branch should gradually receive:

- polished abstract
- 1-page introduction
- related work notes with citations
- problem formulation
- Phase 1 method and annotation protocol
- Phase 1 result tables and figures
- limitations and threat-to-validity section

Do not move selected figures into the paper branch until the corresponding experiment run is accepted as the current evidence version.
