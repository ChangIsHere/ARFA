# Phase 1 Residual Validation

## Research Question

Can the gap between expected terminal outcome and actual terminal observation predict whether a coding agent needs to reason again?

## Dataset Construction

The first dataset is deterministic and synthetic. Each record represents one terminal execution step with an instruction, action, expectation, observation, exit code, binary reasoning-necessity label, and label rationale.

The initial dataset includes successful and failed examples for tests, syntax checks, missing files, wrong directories, shell commands, runtime exceptions, package imports, and empty outputs.

## Label Definition

`reasoning_needed = true` when the terminal output blocks safe continuation of the current plan, contradicts the expectation, or reveals failure.

`reasoning_needed = false` when the output confirms expected progress and the next planned action remains valid.

## Residual Definitions

Embedding residual:

```text
1 - cosine_similarity(expectation, observation)
```

The implementation uses `sentence-transformers/all-MiniLM-L6-v2` when available. If it is not installed locally, it uses a deterministic TF-IDF fallback so the Phase 1 pipeline remains runnable without network access.

Rule residual uses exit code, error keywords, traceback patterns, test failure patterns, missing-file signals, and expectation-observation contradiction cues.

Hybrid residual:

```text
0.6 * embedding_residual + 0.4 * rule_residual
```

Weights and thresholds are configurable in `config/phase1_residual.yaml`.

## Evaluation Metrics

The pipeline reports accuracy, precision, recall, F1, false-fast rate, false-slow rate, ROC-AUC when available, residual distributions, confusion matrix, and false-fast cases.

## Results

Run:

```bash
bash scripts/phase1_build_dataset.sh
bash scripts/phase1_run_analysis.sh
```

The generated report is written to `results/phase1_residual/phase1_report.md`.

## Failure Cases

False-fast cases are the most important manual-inspection cases because the system incorrectly predicts that reasoning is unnecessary when reasoning is actually needed.

The pipeline writes them to `results/phase1_residual/false_fast_cases.jsonl`.

## Conclusion

The conclusion is intentionally generated from Phase 1 metrics. Do not use residual thresholds for Phase 3 until Phase 1 results are reviewed.
