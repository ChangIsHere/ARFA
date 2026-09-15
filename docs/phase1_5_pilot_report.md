# Phase 1.5 Collection Pilot

## Status

The 10-task engineering pilot passed the collection-readiness gate. It is not a residual-accuracy result and must not be cited as evidence that ARFA works.

## Collection Results

- Tasks completed: 10/10
- Executed terminal steps: 25
- Content-deduplicated annotation items: 23
- Structured-output parse errors: 0
- Complete pre-execution commitments: 100%
- Mean expectation quality: 0.871
- Self-contained current actions: 92%
- Self-contained committed continuations: 100%
- Provisional shadow fast candidates: 7/25 (28%)

The two non-self-contained current actions were caught by the shared action-safety gate. This gate is also applied to the no-expectation baseline and therefore cannot be credited as residual value.

## Audit Outcome

The collector, native Ollama JSON-schema interface, content deduplication, blind packet builder, and incomplete-label refusal are working. The analyzer correctly emits no accuracy, AUC, or routing claim while either annotation packet is incomplete.

The tracked redacted evidence bundle is in `evidence/phase1_5_pilot/`. It contains the collection audit, unlabeled blind packet, redacted 25-step source index, record counts, and SHA-256 metadata. Raw traces and the private source map remain local.

## Required Before Full Collection

Independently label the 23-item primary pilot packet and the frozen 6-item secondary packet using `docs/phase1_5_annotation_guideline.md`. Use the pilot only to resolve guideline ambiguity and verify annotator agreement. Do not tune the residual representation or acceptance thresholds on pilot labels.

After this annotation check, run the frozen three-model shadow matrix over the 100 Phase 1.5 tasks. The separate 100-task `phase3_final` split remains excluded from all residual fitting and threshold selection.
