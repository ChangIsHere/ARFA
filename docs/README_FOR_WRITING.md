# ARFA Workshop Paper Handoff

This is the canonical entry point for drafting the workshop paper. Read this
file first, then use [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) for exact
numbers and [`evidence/`](../evidence/README.md) for source artifacts.

For a new writing session, provide these three files together. Treat the result
tables as the numerical source of truth, keep development and reserved-final
results separate, and do not infer unreported significance or energy savings.

## Working Title

**ARFA: Residual-Guided Model Selection for Terminal-Based Coding Agents**

## Research Question

After a terminal action, can the difference between the expected execution
outcome and the actual terminal observation help select the model for the next
reasoning turn?

## Method In One Paragraph

Before executing a command, the agent commits to an expected outcome, expected
exit code, stdout/stderr presence, and a possible continuation. After execution,
ARFA computes a structured residual from mismatches between those commitments
and the terminal observation. A low residual routes the next turn to a fast
model; a high residual or a failed safety check routes it to a slow model. Both
models use the same agent loop, evaluator, context limit, and step cap.

## Paper Thesis

The result is conditional rather than universal:

> Execution residual can guide fast/slow model selection in terminal agents,
> but the gain depends on the model pair. Residual works best when the fast model
> is capable enough to avoid retries and genuinely cheaper over the complete
> trajectory.

This framing includes both the positive and negative findings. Gemma and the
Qwen 7B/Gemma 12B mixed pair show clear benefits. Llama and Phi explain why a
nominally smaller model can increase total cost through extra calls, repeated
actions, or parse failures.

## Contributions

1. **Execution residual for model selection.** ARFA uses a pre-execution
   commitment and the resulting terminal observation to select the next model.
2. **Matched online evaluation.** Residual is compared with always-slow,
   always-fast, learned observation-only, expectation-gate-only, and
   budget-matched random routing in the same execution loop.
3. **Model-pair analysis.** Four native pairs and two cross-family pairs show
   that routing quality and model choice must be evaluated together.
4. **Cost-reversal finding.** A smaller model can cost more over a full task when
   it creates extra reasoning steps or execution errors.
5. **Cross-model state transfer.** A compact structured handoff retains the
   measured success level while using less time and fewer tokens than full
   visible history.

## Evidence Chain

### 1. Offline signal

The shadow collection contains 300 trajectories and 790 terminal steps. Full
residual reaches ROC-AUC 0.743 against reviewed reasoning-demand labels. Learned
observation-only reaches 0.749, so the offline study establishes association but
not superiority.

### 2. Standalone model behavior

Eight models each run the same 200 tasks. This stage identifies candidate pairs
and shows that model size does not directly predict total token use or task
success. It is characterization, not the primary routing comparison.

### 3. Four native development pairs

Residual reduces slow-model calls in all four pairs. Qwen and Gemma reduce total
cost; Llama and Phi increase time or tokens because the fast model creates more
work. These development results motivate pair-dependent analysis.

### 4. Reserved Qwen and Gemma evaluation

Each pair runs six policies on the same reserved 100 tasks. The strongest result
is Gemma residual versus always-slow: +9 success points, -21.5% wall time,
-12.7% tokens, and -43.5% slow calls (`p=0.0225`). Qwen preserves always-slow
success within 1 point while reducing wall time by 14.4% and slow calls by 41.3%.

### 5. Cross-family model selection

Qwen 7B/Gemma 12B improves success by 4 points over always-slow while reducing
wall time by 33.5%, tokens by 25.8%, and slow calls by 49.5%. Reversing the
families to Gemma 4B/Qwen 14B saves time but loses 3 success points. The
interaction therefore depends on both model roles, not only parameter count.

### 6. Handoff behavior

Three modes are evaluated: full visible history, structured state, and latest
observation only. Structured state differs from full history by -1 success point
while reducing wall time and tokens by about 21%. Observation-only requires
28.2% more calls and 28.3% more slow calls than structured state.

Exact values and comparisons are in
[EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md).

## Recommended Workshop Structure

1. **Introduction**
   - Terminal agents repeatedly invoke reasoning models after tool execution.
   - Existing efficiency work often optimizes inference after a call is chosen.
   - ARFA asks which model should handle the next call based on execution.
2. **Related Work**
   - Coding agents and ReAct-style loops.
   - Reflection and execution feedback.
   - Model routing and adaptive computation.
   - Inference and agent-efficiency measurement.
3. **Method**
   - Pre-execution commitment.
   - Structured residual formula.
   - Fast/slow routing and safety fallback.
   - Cross-model handoff state.
4. **Experimental Setup**
   - InterCode NL2Bash and the 100/100 task split.
   - Eight models and six model pairs.
   - Matched policies and success evaluator.
   - Task-level bootstrap intervals and exact McNemar tests.
5. **Results**
   - Offline residual association.
   - Native-pair development study.
   - Reserved Qwen and Gemma matrices.
   - Cross-family pair analysis.
6. **Analysis**
   - Fast-model cost reversal.
   - Handoff ablation.
   - When residual helps and when it does not.
7. **Limitations**
   - One task benchmark.
   - One run per task and policy.
   - Offline labels are provisional reviewed annotations.
   - Current full-matrix GPU measurements are not policy-level causal energy
     estimates.
8. **Conclusion**

## Claims Supported By The Data

- Residual can reduce slow-model use in a matched terminal-agent loop.
- Some pairs preserve or improve success while reducing time and tokens.
- The benefit is model-pair dependent.
- A smaller model can increase complete-trajectory cost.
- Structured handoff can reduce context cost without a measured success loss
  distinguishable at the current sample size.

## Claims To Avoid

- Residual universally outperforms learned observation-only routing.
- Every fast/slow pair reduces total cost.
- Phase 2 standalone runs are causal routing baselines.
- Absolute latency can be pooled across different hardware classes.
- Current run-level GPU samples prove policy-level energy savings.
- The current results establish transfer beyond InterCode.

## Terminology

- **Success rate:** fraction of tasks accepted by the task evaluator. It is not
  command-match accuracy.
- **Residual:** mismatch between the committed expected execution signals and the
  observed terminal result.
- **Fast-call rate:** fraction of reasoning calls assigned to the fast model.
- **Slow-call reduction:** reduction in slow-model calls relative to the stated
  comparator, normally always-slow.
- **Development:** the 100-task split used for pair and threshold development.
- **Reserved final:** the separate 100-task split used by the frozen Qwen and
  Gemma matrices and the mixed-pair extension.

## Source Map

- Residual formula: `src/phase1_5_shadow/residual.py`
- Online controller: `src/phase3_arfa/arfa_agent.py`
- Router policies: `src/phase3_arfa/router.py`
- Agent prompts: `src/phase3_arfa/prompts.py`
- Paired analysis: `src/phase3_arfa/analyze_results.py`
- Handoff analysis: `src/phase3_arfa/analyze_handoff.py`
- Frozen task data: `data/phase2/intercode_nl2bash_official_200.jsonl`
- Phase 2 tables and figures: `docs/phase2_writing_bundle/`
- Native-pair evidence: `evidence/phase3_design/native_pairs/`
- Final Qwen/Gemma evidence: `evidence/phase3_final/`
- Mixed-pair evidence: `evidence/phase3_mixed_pairs/`
- Handoff evidence: `evidence/phase3_handoff/`

## Remaining Work Beyond The Workshop Draft

The next high-value experiment is direct transfer to a pinned external terminal
benchmark without retuning the routing threshold. Repeated policy-level energy
measurement and residual-component ablations follow. See
[phase3_publication_experiment_plan.md](phase3_publication_experiment_plan.md).
