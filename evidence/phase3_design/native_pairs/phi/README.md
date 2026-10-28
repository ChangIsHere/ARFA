# Phi Phase 3 Development Study

This bundle records the Phi development study. Phi-4 Mini 3.8B and Phi-4 14B
were resident together on one NVIDIA L40 48GB GPU. All policies used the same
100 InterCode NL2Bash development tasks.

No tested residual threshold met the saved rule requiring at least slow-only
success with lower wall time and fewer slow calls. The `0.15` result below had
the highest development success and is retained for diagnosis; it was not a
selected operating point and is not held-out evidence.

## Results

- Residual at 0.15: 33/100 successful, 161.90 seconds/task, 214 slow-model calls.
- Always-slow: 30/100 successful, 147.40 seconds/task, 299 slow-model calls.
- Always-fast: 23/100 successful, 176.65 seconds/task, no slow-model calls.

Against always-slow, residual reduced slow-model calls by 28.43%, but increased
wall time by 9.83% and tokens by 2.47%. The success-rate difference was 3 points
with a task-bootstrap 95% interval of [-3, 9] points. This pair shows that fewer
slow calls alone do not guarantee lower end-to-end cost.

`analysis.json` and `policy_summary.csv` contain the paired analysis. Raw traces
remain in the ignored results directory and are linked by `manifest.json`.
The three `*_metrics.csv` files contain the 100 task-level records for each
policy.
