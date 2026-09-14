# Phase 2 Writing Bundle

This directory is the compact, tracked evidence package for writing the ARFA workshop paper's Phase 2 baseline section.
Phase 2 is a standard always-reason structured ReAct agent. It contains no residual gate and no fast/slow routing.

## Evidence Status

- Complete matrix: 3 local models x 200 tasks = 600 runs.
- Main set: the full released InterCode NL2Bash suite, used as the ARFA Phase 2 evaluation set; it is not presented as an upstream official test split.
- Environment audit: 200/200 tasks executed, 199 reached released-gold self-success, and 195 met the stricter gold-healthy criterion.
- Main results use all 200 tasks. A 195-task gold-healthy sensitivity analysis is reported separately.
- Completeness gate: PASS.

## Main Results

- Qwen2.5-Coder 7B: 52/200 = 26.0%; 5.28 calls/task; 6429.4 tokens/task; 47.29 s/task.
- Llama 3.1 8B: 51/200 = 25.5%; 3.98 calls/task; 3707.7 tokens/task; 58.69 s/task.
- Qwen2.5-Coder 14B: 80/200 = 40.0%; 3.29 calls/task; 3305.1 tokens/task; 85.70 s/task.

## Paired Findings

- Qwen 14B exceeds Qwen 7B by 14.0%; paired bootstrap 95% CI [7.5%, 20.5%], exact McNemar p=6.17e-05.
- Qwen 14B exceeds Llama 8B by 14.5%; paired bootstrap 95% CI [8.0%, 21.0%], exact McNemar p=3.85e-05.
- Qwen 7B and Llama 8B differ by only 0.5 percentage points; the paired CI includes zero and McNemar p=1.0.

## Protocol

- Runtime: Ollama 0.34.0 on Apple M4 with 24 GB unified memory; models run serially and locally.
- Models: Qwen2.5-Coder 7B Q4_K_M, Llama 3.1 8B Q4_K_M, and Qwen2.5-Coder 14B Q4_K_M.
- Shared effective context: 8192 tokens; temperature 0.0; maximum 700 generated tokens per call; maximum 12 ReAct steps.
- Agent protocol: structured zero-shot ReAct with action, expected outcome, observation, and an explicit done signal.
- Environment: fresh agent and evaluator Docker containers per task, with the released InterCode reward reimplemented and pinned.
- Success: reward >= 0.99.

## Interpretation Boundaries

- Phase 2 establishes the always-reason baseline only; it does not test ARFA's residual-guided savings yet.
- There is one run per model. Wilson intervals and paired bootstrap intervals quantify variation across tasks, not across random seeds.
- Local Q4 quantization, hardware, Ollama version, prompt format, and the 8K context are part of the experimental condition.
- Token totals are server-reported cumulative prompt-plus-completion tokens across calls, not API billing tokens.
- Five released tasks have gold/environment anomalies. They remain in the 200-task main result and are removed only in the labeled sensitivity analysis.
- Failure categories are deterministic trace-derived diagnostics, not manually adjudicated causal labels.
- The copy-or-move category has only four tasks and zero successes for all models; subgroup claims should acknowledge the small sample.

## Artifact Map

- `reports/phase2_report.md`: complete main, paired, diagnostic, sensitivity, and completeness report.
- `reports/failure_case_analysis.md`: failure counts and representative cases.
- `metrics/main_results.csv`: all-200 aggregate metrics with Wilson intervals.
- `metrics/main_results_gold_healthy.csv`: 195-task sensitivity analysis.
- `metrics/paired_model_comparisons.csv`: paired bootstrap intervals and exact McNemar tests.
- `metrics/results_by_filesystem.csv` and `metrics/results_by_task_category.csv`: subgroup results.
- `metrics/failure_cases.jsonl` and `metrics/failure_counts.csv`: trace-derived failure diagnostics.
- `metrics/gold_validation_anomalies.jsonl`: the five tasks excluded from the sensitivity analysis.
- `figures/`: paper-facing PNG and PDF figures.
- `tables/`: Markdown and LaTeX tables.
- `protocol/`: frozen config, model matrix, environment summary, and per-model summaries.
- `protocol/source_artifact_manifest.json`: SHA-256 hashes and sizes for full local traces.

## Recommended Paper Claim

Under a fixed local structured-ReAct protocol, the 14B coding model improves task success over both 7B/8B baselines, but incurs the highest wall-clock latency. These Phase 2 measurements provide the paired always-reason reference required for evaluating whether Phase 3 ARFA reduces reasoning calls, tokens, and latency without materially reducing success.
