# Gemma Phase 3 Development Study

This bundle records the selected Gemma development result. Gemma 3 4B and 12B
were resident together on one NVIDIA L40 48GB GPU. All policies used the same
100 InterCode NL2Bash development tasks.

The residual threshold was selected on these development tasks. Of the tested
values, `0.30` minimized wall time per successful task while meeting the saved
development rule. It is therefore a tuning result, not held-out evidence.

## Results

- Residual: 29/100 successful, 241.31 seconds/task, 201 slow-model calls.
- Always-slow: 26/100 successful, 347.96 seconds/task, 387 slow-model calls.
- Always-fast: 21/100 successful, 178.07 seconds/task, no slow-model calls.

Against always-slow, residual reduced slow-model calls by 48.06%, wall time by
30.65%, and tokens by 30.61%. Their task-bootstrap 95% intervals were
[38.78%, 55.71%], [19.17%, 40.17%], and [12.73%, 43.94%]. The success-rate
difference was 3 points with a 95% interval of [-3, 9] points.

`analysis.json` and `policy_summary.csv` contain the paired analysis. Raw traces
remain in the ignored results directory and are linked by `manifest.json`.
The three `*_metrics.csv` files contain the 100 task-level records for each
policy.
