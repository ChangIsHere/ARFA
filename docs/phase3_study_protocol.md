# Phase 3 Qwen Study: Decision Rules Before Formal Runs

**Design version:** `phase3-qwen-pair-v1`, in
[`config/phase3_study.yaml`](../config/phase3_study.yaml). This document fixes
the proposed decision rules before collecting Phase 3 formal outcomes. Phase 3
still needs an implemented controller, a matched baseline, and a runnable
measurement setup. A later design change should get a new version and be reported
as a change, not presented as part of this version.

## Why Five Percentage Points?

On the full 200-task Phase 2 benchmark, Qwen 14B succeeded on 80 tasks (40%) and
Qwen 7B on 52 (26%). The gap is **14 percentage points**. We set the maximum
acceptable loss against a matched 14B-only agent at **5 absolute percentage
points**, equivalent to 5 of the 100 reserved test tasks. At the observed Phase 2
rates, that would retain 9 of the 14 points separating the two models, about 64%
of the observed advantage. Five points is therefore a substantive limit on how
much quality we would trade for efficiency. It is a project choice, not a
universal benchmark standard or a margin inferred from Phase 3 outcomes.

Phase 2's 14B-vs-7B task-paired bootstrap interval for their 14-point gap is
7.5-20.5 points. This supports treating that gap as substantial, but it does not
prove that a five-point noninferiority claim will be testable with 100 tasks. On
the reserved 100 tasks, the saved Phase 2 outcomes are 43/100 for 14B and 30/100
for 7B. Those tasks were already run in Phase 2; they are reserved from residual
fitting, not globally unseen.

## Data And Comparisons

Use exactly the 200 task IDs in the Phase 2 InterCode NL2Bash manifest and the
existing 100/100 Phase 1.5 split. Development uses only the 100 shadow tasks.
Choose the router threshold and continuation policy there. Fix all prompts,
model hashes, max steps, threshold, recovery rules, and analysis code before
running the `phase3_final` tasks. The primary analysis uses those 100 tasks.
The full 200-task summary is secondary because half were used in development.

The primary comparison is ARFA (Qwen 14B slow + Qwen 7B fast) against an
**always-14B agent run under the same Phase 3 prompt, environment, and budget**.
The original Phase 2 14B result is a historical reference, not the matched
comparator. Also run 7B-only, a fixed-switch policy, and an observation-only
router under the same protocol. Compare identical task IDs and report failures
and recovery, not only aggregate success.

## Prespecified Conclusions

For each reserved task, let `d_i = success_ARFA_i - success_14B_i`. Estimate the
mean paired difference with a task-clustered bootstrap (10,000 draws, seed 1507).
At one run per policy/task, resample task pairs; if formal repeats are added,
resample tasks with all their repeat observations together. Report the two-sided
95% percentile interval. Claim success noninferiority only when its lower bound
is **above -0.05**. Also give the point estimate, task counts, and discordant
task outcomes. Failure to clear the boundary means *inconclusive* unless the
interval is entirely below it; a point estimate within five points alone is not
proof. This is an adapted noninferiority analysis for agent outcomes, not a
clinical-trial claim.

For efficiency, use total wall seconds **per task**, including model calls,
scoring, retries, model switching, and command execution. Compute the percent
reduction in mean paired-task time relative to matched 14B-only. Our practical
target is at least **15%** with a bootstrap 95% interval whose lower bound is
above zero. A similar or better success rate with slower total execution would
not establish an efficiency benefit. Report the total 14B-call reduction as a
separate target of 25%; all 7B calls and tokens still count. Energy is reported
only if actual measurement works on this Mac and covers the same task window.

To show value beyond choosing a smaller model, compare ARFA success against
7B-only. To attribute a gain to residual, compare against observation-only
routing with the same fast/slow models and matched runtime. Neither comparison
is replaced by the historical Phase 2 baseline.

## Precision Check Before Spending Compute

The five-point margin must not be widened after seeing Phase 3 outcomes. With
100 test tasks, precision depends strongly on how often ARFA and matched 14B
disagree. For illustration, even if their observed success rates are identical,
10 discordant tasks out of 100 give an approximate 95% half-width of 6.2 points
(`1.96 * sqrt(0.10 / 100)`). This is only a planning calculation; the actual
paired bootstrap interval will use observed outcomes. A 100-task test can thus
support a clear estimate yet still leave a five-point conclusion unresolved.
Repeated runs help measure run-to-run variation but do not create new task IDs.
If precision is inadequate, report the result as inconclusive and seek additional
independent tasks for a stronger claim; keep the original criterion visible.

Methodological reference: the [CONSORT noninferiority extension](https://www.equator-network.org/reporting-guidelines/consort-non-inferiority/)
recommends stating the margin, its rationale, and an interval-based decision in
advance. We borrow that reporting discipline; the 5-point value is specific to
this agent experiment.
