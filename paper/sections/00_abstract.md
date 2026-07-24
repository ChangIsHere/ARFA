# Abstract

Terminal-based coding agents often alternate between reasoning, command execution, and observation. Existing efficiency methods usually optimize how reasoning is performed, cached, routed, or compressed. This work studies a complementary question: after an execution step, is renewed reasoning necessary at all?

We introduce execution residual as a candidate signal: the discrepancy between a pre-execution expected terminal outcome and the actual terminal observation. In Phase 1, we evaluate whether residual scores predict reasoning necessity on external InterCode Bash trajectories under task-level held-out splits and safety-constrained threshold selection.

This draft is preliminary. Final claims require human-reviewed labels and a rerun with the approved embedding model.
