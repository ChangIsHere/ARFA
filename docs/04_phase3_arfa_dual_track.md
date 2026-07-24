# Phase 3 ARFA Dual Track

Phase 3 is locked until Phase 1 and Phase 2 are complete and reviewed.

Planned ARFA design:

```text
if safety_trigger:
    slow_track
elif residual > threshold:
    slow_track
elif valid_next_plan_exists:
    fast_track
else:
    slow_track
```

The residual threshold must come from Phase 1 analysis. It should not be chosen arbitrarily.
