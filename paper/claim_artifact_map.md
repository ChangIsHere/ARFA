# Claim-Artifact Map

| Paper Claim | Required Artifact | Current Status |
| --- | --- | --- |
| The 50-record pilot only validates the pipeline. | `results/phase1_residual/pilot_50/` | Local-only, ignored by git |
| Final Phase 1 uses external terminal trajectories. | `data/phase1/external_intercode/` | Local external clone, ignored by git |
| Final Phase 1 uses task-level splits. | `results/phase1_residual/final/splits.json` | Local generated artifact |
| Thresholds are selected on validation only. | `results/phase1_residual/final/thresholds.json` | Local generated artifact |
| Held-out test metrics do not pass the evidence gate yet. | `results/phase1_residual/final/test_metrics.csv` and `phase1_final_report.md` | Local generated artifact |
| Paper figures are generated from final Phase 1 outputs. | `scripts/phase1_make_paper_artifacts.sh` | Code tracked on `main` |

Before submission, copy selected final figures/tables into this branch only after rerunning the final experiment with the approved embedding backend and reviewed labels.
