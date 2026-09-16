# Results Workspace

Generated experiment outputs are written here and are ignored by git by default.

Current residual diagnostics combine the existing Phase 1.5 shadow traces. Export
their numeric evidence without rerunning agents:

```bash
bash scripts/phase1_export_diagnostics.sh
```

The export is tracked under `evidence/residual_diagnostics/`. Original traces stay
under `phase1_5_shadow/full/`; existing labels and analysis stay at their source
paths. `phase1_residual/` holds the supplementary external-trajectory study and
earlier pilot outputs. None of these historical results is silently overwritten.

`phase2_baseline/` remains unchanged; its tracked compact package is
`docs/phase2_writing_bundle/`. See the root README for the current writing order.
