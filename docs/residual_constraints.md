# Study Notes

These notes describe how to interpret the saved results. They do not prevent
building and testing the Phase 3 agent.

## Scores And Task Outcomes

The original Phase 1 constructed expectations from downloaded trajectories after
collection. Its heuristic labels can overlap with scoring rules, so it is now a
supplementary study. Phase 1.5 recorded expectations before execution, but always
called the reasoning model afterward. Its fast decisions were suggestions only.

A small residual can accompany a wrong plan or an unusable next command. The
current command checker is a string heuristic and can miss missing stdin in a
pipeline. All four Phase 3 development pairs measure actual execution and cost;
the reserved final split is still unused.

Full residual AUC was 0.743 against the reviewed labels, versus 0.749 for learned
observation-only scoring. The selected semantic weight was zero. Task-outcome
associations vary across models; whole-trajectory averages also use later steps
and should not be described as predictions available at the first decision.

## Labels

The project owner reports that six people divided and reviewed AI-assisted labels.
Separate reviewer files are not available. Stored agreement numbers compare AI
judges, not six human annotators. The earlier 23+6 human pilot is a separate check
of the labeling instructions.

The original AI judge prompt did not explain its short output codes. This has
been corrected for future runs, but its effect on the reviewed labels has not
been measured. Existing label-based results remain provisional. Original labels
and audits are unchanged. A documented human-adjudicated subset or a new analysis
with the corrected judge would help assess this issue.

The new task-outcome comparisons use evaluator outcomes rather than subjective
labels. The saved fusion weight still comes from the earlier label-based analysis.

## Earlier Selection Rule

The first offline selection rule found no operating point meeting all its
requirements. That rule remains in the frozen protocol and saved analysis. The
paper relies on matched online policies rather than treating the offline rule as
a deployment test.

Original references: `config/phase1_5_shadow.yaml` and
`results/phase1_5_shadow/analysis_ai_blind/analysis.json`. Current development
uses `config/phase3_arfa.yaml`.

Phase 2 covers eight locally quantized models on one benchmark, with one run per
task/model. Its intervals describe variation across tasks, not repeated runs.
Reserved Phase 3 tasks were excluded from residual fitting but were already run
in Phase 2; they are not entirely unseen tasks.
