# Future Phases

Phase 2 will build a standard ReAct terminal coding-agent baseline that always reasons after each observation.

Phase 3 will build ARFA as a dual-track controller:

- fast track: continue the existing plan without a large-model reasoning call
- slow track: invoke the reasoning model and re-plan

The residual threshold used in Phase 3 must be selected from reviewed Phase 1 validation results.
