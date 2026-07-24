# ARFA Paper Branch

This branch stores manuscript text and writing assets for:

**ARFA: Residual-Guided Reasoning Control for Terminal-Based Coding Agents**

The code branch is `main`. Paper claims should be linked to concrete scripts, configs, and generated artifacts from `main`.

## Structure

```text
paper/
├── README.md
├── outline.md
├── claim_artifact_map.md
├── sections/
│   ├── 00_abstract.md
│   ├── 01_introduction.md
│   ├── 02_related_work.md
│   ├── 03_problem_formulation.md
│   ├── 04_phase1_method.md
│   ├── 05_phase1_results.md
│   ├── 06_limitations.md
│   └── 07_future_phases.md
└── notes/
    └── writing_todos.md
```

## Current Evidence Status

Phase 1 uses external InterCode Bash trajectories. The current result is preliminary because the semantic backend may still be `tfidf_fallback`; rerun with `sentence-transformers/all-MiniLM-L6-v2` before making final embedding claims.

Phase 2 and Phase 3 remain locked until Phase 1 passes the evidence gate.
