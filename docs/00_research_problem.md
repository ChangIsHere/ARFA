# Research Problem

Terminal-based coding agents edit files, run commands, observe terminal outputs, and decide what to do next.

Most efficiency methods optimize reasoning after reasoning has already been invoked: caching, memory, model routing, context compression, or planning. ARFA focuses on a different question:

> After the current terminal execution, is another expensive reasoning step actually necessary?

The working hypothesis is that the execution residual, defined as the discrepancy between the expected terminal outcome and the actual terminal observation, can serve as a lightweight signal for this decision.

This repository uses a phased implementation discipline:

1. Validate whether residual predicts reasoning necessity.
2. Build a standard ReAct terminal-agent baseline.
3. Build ARFA as a residual-guided dual-track agent.

Phase 2 must not start until Phase 1 produces a dataset, metrics, failure-case analysis, and written report.
