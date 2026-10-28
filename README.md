# ARFA

**Residual-Guided Model Selection for Terminal-Based Coding Agents**

ARFA asks a narrow question: after a terminal action, can the difference between
the expected and observed result help choose the model for the next reasoning
turn? The agent records an expectation before execution, computes a structured
execution residual afterward, and routes the next call to a fast or slow model.

## Evidence Snapshot

- **Offline diagnostics:** 300 shadow trajectories, 100 shared InterCode
  NL2Bash tasks, three models, and 790 executed steps.
- **Standalone characterization:** eight models each ran the same 200 tasks,
  for 1,600 Phase 2 attempts.
- **Online development study:** four fast/slow pairs each ran 100 shared tasks
  under residual, always-slow, and always-fast policies, for 1,200 matched
  policy-task attempts.

| Pair | Residual | Slow-only | Fast-only | Residual versus slow-only |
| --- | ---: | ---: | ---: | --- |
| Qwen 7B/14B | 41% | 41% | 31% | 40.9% fewer slow calls; wall-time CI crosses zero |
| Llama 3B/8B | 19% | 16% | 8% | 22.6% fewer slow calls; 13.8% more wall time |
| Gemma 4B/12B | 29% | 26% | 21% | 48.1% fewer slow calls; 30.6% less wall time and tokens |
| Phi Mini/14B | 33% | 30% | 23% | 28.4% fewer slow calls; 9.8% more wall time |

These are development results. Gemma's threshold was selected on the same
development tasks. No Phi threshold passed the saved selection rule, so its
best-success result is diagnostic. The reserved Phase 3 final split remains
unused.

The main finding is therefore conditional: residual routing can reduce slow
model use, but end-to-end savings depend on the model pair. Qwen preserved
slow-only success while reducing slow calls. Gemma produced the clearest
development tradeoff. Llama and Phi show that a smaller model can add latency,
tokens, or retries even when routing invokes the larger model less often.

Trace analysis explains part of this reversal. Relative to slow-only, the
Llama fast model used 26.6% more calls per task and 23.4% more tokens per call;
the Phi fast model used 37.5% more calls and produced more repeated actions and
parse errors. A smaller model is therefore not automatically a cheaper route.

## Routing Controls

The Gemma 4B/12B control study is complete on the 100-task development split.
Residual routing completed 28 tasks, compared with 25 for expectation-gate-only
and 25 for budget-matched random routing, while using fewer tokens and model
calls. Learned observation-only completed 31 tasks with similar token use.
These differences are not statistically significant at the current sample
size. The result supports residual over simple routing controls, but not over
the learned observation baseline.

The next paper-facing step is a frozen evaluation on the reserved tasks,
followed by broader mixed-pair coverage and a second benchmark. See
[the control study](docs/phase3_control_extension.md).

## Writing And Evidence

Start with [the manuscript handoff](docs/README_FOR_WRITING.md). It lists the
supported claims, exact result sources, paper outline, and statements that
should remain qualified.

- [Residual definitions and offline diagnostics](docs/residual_diagnostics.md)
- [Eight-model Phase 2 bundle](docs/phase2_writing_bundle/README_FOR_WRITING.md)
- [Four-pair Phase 3 evidence](evidence/phase3_design/README.md)
- [Four native-pair results](evidence/phase3_design/native_pairs/results.csv)
- [Gemma routing-control evidence](evidence/phase3_design/supplementary/gemma_controls/README.md)
- [Online experiment design](docs/04_phase3_arfa_dual_track.md)
- [Next experiments for a full paper](docs/phase3_publication_experiment_plan.md)
- [Gemma routing controls and follow-up experiments](docs/phase3_control_extension.md)

## Repository Map

- `src/`: agents, scoring, runners, and analysis.
- `config/`: experiment configurations.
- `deployment/nrp/`: bounded NRP GPU jobs used by the completed runs.
- `data/phase2/`: frozen 200-task InterCode manifest.
- `evidence/`: compact results, confidence intervals, and provenance.
- `docs/`: writing handoff, protocols, tables, and figures.
- `results/`: ignored raw traces and checkpoints kept locally.

## Verification

```bash
.venv/bin/python -m pytest -q
bash -n scripts/*.sh deployment/nrp/*.sh
git diff --check
```

Tasks come from [InterCode](https://github.com/princeton-nlp/intercode). Model
weights, caches, raw traces, credentials, kubeconfig, and tokens are not
committed.
