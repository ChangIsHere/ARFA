# ARFA Research Memory

This document records the stable project framing that should guide future experiments and writing.

## Project Name

ARFA = **Action Residual Fused Agent**.

Use this expansion when the full name is needed:

> We introduce ARFA (Action Residual Fused Agent), a residual-guided reasoning-control framework for terminal-based coding agents.

The working paper title remains:

> ARFA: Residual-Guided Reasoning Control for Terminal-Based Coding Agents

Do not replace this with other expansions unless the project name is explicitly redefined.

## Core Research Question

After a terminal-based coding agent executes an action and receives terminal feedback, does it always need to call an expensive reasoning model again?

ARFA studies the decision before reasoning:

> Is renewed reasoning necessary after the current execution step?

This is different from caching, memory, routing, or compression, which usually optimize reasoning after the system has already decided to reason.

## Core Signal

The central signal is execution residual:

```text
residual = discrepancy(expected execution outcome, actual terminal observation)
```

Interpretation:

- small residual: the terminal output matches expectation, the current plan may remain valid, and the agent may skip expensive reasoning
- large residual: execution deviated from expectation, the plan may be invalid, and the agent should reason or replan

The theoretical framing should emphasize that residual is an external execution-state signal, not merely an internal confidence estimate.

## AI Agent Framing

The project should be framed as an AI agent systems project, not simply as a classifier project.

Important keywords:

- LLM agents
- coding agents
- agent orchestration
- tool use
- terminal environment
- planning
- execution feedback
- reasoning control
- fast-slow routing
- agent evaluation
- AI infrastructure

## Current Truthful Status

Completed or partially implemented:

- Phase 1 repository scaffold
- pilot 50-record local sanity check
- external InterCode Bash data ingestion
- expectation-observation logging schema
- residual scoring pipeline
- validation-selected thresholds
- held-out Phase 1 analysis
- safety metrics including false-fast and false-slow rates
- subgroup, ablation, leakage-audit, and paper-figure generation scripts
- GitHub repository with `main` and `paper` branches

Not yet completed:

- human-reviewed final annotation
- final MiniLM/SentenceTransformer embedding run
- ARFA-Min as a live agent controller
- ReAct baseline execution against the same tasks
- token, latency, and large-model-call reduction results
- SWE-bench Lite validation
- workshop-ready final experiment section

Do not claim large token or latency reductions until Phase 2/3 experiments actually produce them.

## Phase Roadmap

Phase 1:

Validate whether execution residual predicts reasoning necessity on external terminal-based trajectories. This phase is about signal validation, not full agent deployment.

Phase 2:

Build a standard terminal-based ReAct baseline that always reasons after each observation.

Phase 3:

Build ARFA-Min with fast-slow routing:

```text
Action + Expected Outcome
-> execute terminal command
-> Actual Terminal Output
-> residual(Expected, Actual)
-> skip reasoning if residual is small and the plan remains safe
-> trigger reasoning if residual is large or safety rules fire
```

Phase 4:

Extend to stronger benchmarks such as SWE-bench Lite after the minimal system is stable.

## Workshop Target

The near-term goal is a workshop-submittable draft. The likely paper shape is:

1. Introduction
2. Related Work
3. Problem Formulation
4. Method
5. Phase 1 / Preliminary Study
6. Experimental Setup
7. Results
8. Discussion
9. Conclusion

For the first workshop version, the paper may focus on the motivation, residual signal validation, and a minimal ARFA-Min prototype if Phase 2/3 results become available in time.

## Related Work Buckets

- LLM-based software engineering agents
- ReAct and tool-using agents
- planning, reflection, and execution feedback
- efficient inference for large reasoning models
- caching, memory, routing, and context compression
- adaptive computation and fast-slow reasoning
- agent benchmark and evaluation frameworks

Known papers to include:

- ReAct: Synergizing Reasoning and Acting in Language Models
- InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback
- SwiftSage
- Efficient Inference for Large Reasoning Models: A Survey

## Writing Discipline

Use careful language:

- say "tests whether residual is useful"
- say "preliminary evidence"
- say "held-out evidence gate"
- say "reasoning-control signal"

Avoid overclaiming:

- do not say residual is proven necessary
- do not claim final ARFA efficiency before running Phase 2/3
- do not describe heuristic labels as human gold labels
- do not describe TF-IDF fallback as MiniLM embedding evidence
