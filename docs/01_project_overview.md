# Project Overview

ARFA asks whether the difference between an expected and an observed terminal
result can help select the model for the next reasoning turn.

The project has four connected evidence layers:

1. **Offline diagnostics.** Shadow runs record expectations before execution and
   measure whether execution residual is associated with reviewed reasoning
   demand and task outcomes.
2. **Standalone characterization.** Eight models run the same 200 InterCode
   NL2Bash tasks to measure capability, latency, token use, and candidate
   fast/slow pairings.
3. **Matched online control.** Residual, endpoint policies, observation-only,
   expectation-only, and random routing run through the same agent loop and
   evaluator.
4. **Pair and handoff analysis.** Native and cross-family model pairs test
   model-selection dependence; three handoff modes test how much explicit state
   must cross a model switch.

All eight standalone runs, four native development pairs, two reserved final
policy matrices, two mixed-pair matrices, and the handoff ablation are complete.
The strongest result is Gemma 4B/12B: on the reserved split, residual improved
success by 9 points over always-slow while reducing wall time by 21.5%, tokens
by 12.7%, and slow calls by 43.5%.

The supported claim is conditional. Residual can improve fast/slow selection,
but the fast model must be capable enough to avoid retries and genuinely cheaper
over a full trajectory. Llama and Phi demonstrate the opposite case: fewer slow
calls did not reduce total cost because the fast model generated extra work.

Phase 2 describes the candidate models. The matched Phase 3 policies provide the
main evidence about routing. See [the experiment summary](EXPERIMENT_RESULTS.md)
and [the study overview](STUDY_OVERVIEW.md).
