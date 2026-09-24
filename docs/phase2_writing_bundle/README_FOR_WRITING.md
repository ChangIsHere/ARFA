# Phase 2 Writing Bundle

Evidence for the Phase 2 always-reason ReAct baselines. No residual routing is enabled here.

## Evidence Status

- Complete matrix: 8 local models x 200 identical tasks = 1600 runs.
- Data: the full released InterCode NL2Bash suite, used as ARFA's evaluation set, not an upstream official test split.
- Gold-healthy sensitivity set: 195 of 200 tasks.
- Task-ID, prompt, agent, environment, and reported-protocol checks: PASS.

## Main Results

- Qwen2.5-Coder 7B: 52/200 = 26.0%; 5.28 calls/task; 6429.4 tokens/task; 47.29 s/task.
- Llama 3.1 8B: 51/200 = 25.5%; 3.98 calls/task; 3707.7 tokens/task; 58.69 s/task.
- Qwen2.5-Coder 14B: 80/200 = 40.0%; 3.29 calls/task; 3305.1 tokens/task; 85.70 s/task.
- Llama 3.2 3B: 36/200 = 18.0%; 6.12 calls/task; 7610.0 tokens/task; 24.37 s/task.
- Gemma 3 4B: 52/200 = 26.0%; 3.02 calls/task; 3955.1 tokens/task; 22.98 s/task.
- Gemma 3 12B: 63/200 = 31.5%; 4.11 calls/task; 5424.0 tokens/task; 86.61 s/task.
- Phi-4 Mini 3.8B: 25/200 = 12.5%; 3.72 calls/task; 5535.1 tokens/task; 26.46 s/task.
- Phi-4 14B: 51/200 = 25.5%; 3.14 calls/task; 4356.9 tokens/task; 94.80 s/task.

## Planned Pair Comparisons

- Qwen2.5-Coder 14B minus Qwen2.5-Coder 7B: +14.0% success (paired 95% CI [+7.5%, +20.5%]); McNemar p=6.17e-05.
- Llama 3.1 8B minus Llama 3.2 3B: +7.5% success (paired 95% CI [+1.0%, +14.0%]); McNemar p=0.0357.
- Gemma 3 12B minus Gemma 3 4B: +5.5% success (paired 95% CI [-1.5%, +12.5%]); McNemar p=0.161.
- Phi-4 14B minus Phi-4 Mini 3.8B: +13.0% success (paired 95% CI [+7.0%, +19.0%]); McNemar p=6.88e-05.

## Protocol And Scope

- Apple M4 with 24 GB unified memory; models run serially in Ollama. Original runs used 0.34.0; expansion runs used 0.34.1.
- Shared 8K context, temperature 0, 700 output tokens per call, and a 12-step cap.
- Fresh Docker agent and evaluator containers per task; success means reward >= 0.99.
- Phi-4 Mini 3.8B and Phi-4 14B provide the fourth within-family scale pair.
- Llama 3.2 3B and Llama 3.1 8B belong to different Llama releases.
- Each model has one run per task. Intervals quantify task variation, not run-to-run variation.
- Phase 2 alone does not show any benefit from residual routing; Phase 3 needs matched policy runs.
- Completion counts all attempted tasks, including failures. Recovered JSON wrappers are separated from hard parse failures.
- McNemar p-values are unadjusted for multiple comparisons. Energy was not measured.
- Token means exclude tasks with missing token usage; those tasks remain in success-rate denominators.
- Two request-error tasks (Gemma 12B and Phi Mini) use a synthetic error record: token usage is unknown and call counts do not capture all retries or prior steps. Their cost totals are incomplete.
- DeepSeek candidates were replaced after observed runtime and protocol failures; see ../03_phase2_baseline_agent.md for selection history.
- Original and expansion config file SHA-256 values differ. The recorded task, prompt, agent, environment, and protocol fields match; see `metrics/completeness.json` for both config hashes.

## Files

- `reports/phase2_report.md`: full results and diagnostics.
- `metrics/`: per-model summaries, paired comparisons, subgroups, failures, and completeness audit.
- `figures/` and `tables/`: paper-facing exhibits.
- `protocol/`: config, model metadata, environment audit, and trace hashes.
- `metrics/task_outcomes.jsonl`: compact outcomes for all 1,600 attempts, enabling paired reanalysis without raw conversations.
