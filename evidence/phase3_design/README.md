# Phase 3 Development Evidence

`native_pairs/` is the paper-facing result set. It contains four model pairs,
three policies per pair, and the same 100 InterCode NL2Bash development tasks.

- `native_pairs/results.csv`: the compact main table.
- `native_pairs/qwen/`: Qwen2.5-Coder 7B/14B at threshold `0.35`.
- `native_pairs/gemma/`: Gemma 3 4B/12B at the development-selected threshold
  `0.30`.
- `native_pairs/llama/`: Llama 3.2 3B/3.1 8B at threshold `0.35`.
- `native_pairs/phi/`: Phi-4 Mini/14B diagnostic at threshold `0.15`.

Each pair directory contains task-level metrics, policy summaries, paired
analysis, runtime metadata, and hashes for the ignored raw traces. Qwen and
Gemma are the positive cases. Llama and Phi show that reducing slow-model calls
does not always reduce total cost.

`supplementary/` contains the Gemma and Qwen routing controls plus the
fast-model cost-reversal analysis. These results are useful for analysis but do
not belong in the native-pair main table. Raw traces remain under the ignored
`results/phase3_arfa/nrp/` directory.

All results here are development results. They predate the frozen Qwen and Gemma
evaluation in [`evidence/phase3_final/`](../phase3_final/). Absolute latency is
compared only within a model pair because the development pairs ran on different
hardware.
