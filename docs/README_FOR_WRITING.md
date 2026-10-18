# ARFA Manuscript Handoff

This is the entry point for a workshop or arXiv draft based on the repository's
current evidence. The data support a paper about residual-guided model selection
and model-pair dependence. They do not yet support a broad claim that ARFA is
universally faster, safer, or more energy efficient.

The [control extension](phase3_control_extension.md) is prepared but has no
online results yet. Do not add its planned controls to the current results
table or describe them as completed experiments.

## Research Question

Terminal agents normally invoke a reasoning model again after every action.
ARFA instead records an expected outcome before execution and compares it with
the terminal observation. The resulting residual is used to choose the fast or
slow model for the next turn.

The paper asks:

1. Does expectation-observation mismatch contain information related to agent
   outcomes and reviewed reasoning-necessity labels?
2. How capable and expensive are candidate fast and slow models by themselves?
3. In a matched online agent, can residual routing retain task success while
   reducing use of the slow model?
4. Why do end-to-end savings differ across model pairs?

## Evidence Inventory

### Offline residual study

Use [residual_diagnostics.md](residual_diagnostics.md) and
[`evidence/residual_diagnostics/`](../evidence/residual_diagnostics/).

- 300 shadow trajectories on 100 shared InterCode NL2Bash tasks.
- Three model conditions and 790 executed terminal steps.
- Expectations were committed before command execution; routing remained in
  shadow mode.
- Full residual AUC against the stored reviewed reasoning labels was 0.743
  (task-bootstrap 95% CI 0.681-0.811), but learned observation-only AUC was
  0.749. This supports association, not superiority or safe deployment.
- The 751 labels were AI-assisted and, according to the project owner, divided
  among six human reviewers. Separate reviewer files are unavailable, so these
  label-based results should be described as provisional reviewed annotations,
  not as a six-annotator agreement study.
- Test-split task-failure AUC for structured residual was 0.613, 0.611, and
  0.597 across Qwen 14B, Qwen 7B, and Llama 8B.

### Phase 2 model characterization

Use the [Phase 2 writing bundle](phase2_writing_bundle/README_FOR_WRITING.md).

- Eight models x the same 200 tasks = 1,600 attempts.
- Success ranged from 12.5% to 40.0%.
- The study identifies four candidate scale pairs and their standalone cost.
- Phase 2 is historical characterization, not the causal routing baseline.

### Phase 3 online development study

Use [`evidence/phase3_design/`](../evidence/phase3_design/) and especially
[`development_summary.csv`](../evidence/phase3_design/development_summary.csv).
Each pair ran the same 100 development tasks under residual, always-slow, and
always-fast policies.

| Pair | Residual | Slow-only | Fast-only | Slow-call change | Wall-time change | Token change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 7B/14B | 41% | 41% | 31% | -40.86% | -4.87% | -1.41% |
| Llama 3B/8B | 19% | 16% | 8% | -22.64% | +13.78% | +24.82% |
| Gemma 4B/12B | 29% | 26% | 21% | -48.06% | -30.65% | -30.61% |
| Phi Mini/14B | 33% | 30% | 23% | -28.43% | +9.83% | +2.47% |

Qwen's wall-time and token intervals cross zero. Gemma's threshold `0.30` was
selected on the development tasks. No Phi threshold passed the saved selection
rule; `0.15` is shown only because it had the highest development success.
Absolute timing should not be pooled across hardware.

## Supported Paper Claim

> Execution residual can be used as a lightweight reference signal for choosing
> between fast and slow reasoning models in terminal agents. Across four
> development pairs it consistently reduced slow-model calls, while end-to-end
> savings depended on the fast model's latency, output length, and action quality.

The strongest positive examples are Qwen and Gemma. Llama and Phi are useful
counterexamples: reducing slow calls is not sufficient to reduce total cost.
This pair-dependent result connects residual routing to model selection rather
than treating the router and model pair as independent choices.

## Claims To Avoid

- Do not call the current Phase 3 results held out; the reserved final split is
  still unused.
- Do not claim that residual beats observation-only routing; that matched online
  control has not been run.
- Do not claim measured energy savings. GPU power samples exist for infrastructure
  monitoring, but the current paper results report calls, tokens, and wall time.
- Do not describe Phase 2 as the primary routing baseline. Phase 3's matched
  always-slow and always-fast policies are the relevant controls.
- Do not claim that every pair improved. Llama and Phi increased total cost
  relative to always-slow.

## Suggested Paper Structure

1. **Introduction:** reasoning after every terminal action is expensive; an
   external expectation-observation mismatch may help decide which model to use.
2. **Related Work:** coding agents, ReAct and reflection, adaptive computation,
   model routing, and inference efficiency.
3. **Method:** pre-execution commitment, structured residual, safety fallback,
   and fast/slow routing.
4. **Offline Diagnostics:** 300 shadow trajectories and residual associations.
5. **Experimental Setup:** InterCode NL2Bash, eight standalone models, four
   online pairs, matched policies, success criterion, tokens, calls, and latency.
6. **Results:** Phase 2 characterization followed by the four Phase 3 pair
   comparisons.
7. **Analysis:** why Qwen/Gemma benefit and Llama/Phi do not; model-pair choice
   is part of the routing design.
8. **Limitations:** one benchmark, one run per task/policy, development-set
   threshold selection, no matched observation-only router, and no direct energy
   result.

## Source Map

- Residual formula: `src/phase1_5_shadow/residual.py`
- Online controller: `src/phase3_arfa/arfa_agent.py` and `router.py`
- Phase 3 paired analysis: `src/phase3_arfa/analyze_results.py`
- Frozen task manifest: `data/phase2/intercode_nl2bash_official_200.jsonl`
- Phase 2 tables/figures: `docs/phase2_writing_bundle/`
- Phase 3 compact evidence: `evidence/phase3_design/`
- Raw local traces: ignored `results/` paths referenced by evidence manifests

For the experiments needed beyond the workshop version, see
[phase3_publication_experiment_plan.md](phase3_publication_experiment_plan.md).
