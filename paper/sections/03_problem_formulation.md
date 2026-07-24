# Problem Formulation

At step `t`, a terminal-based coding agent has:

- task context
- current plan
- action `a_t`
- expected terminal outcome `e_t`
- actual terminal observation `o_t`

The Phase 1 prediction target is whether renewed reasoning is needed before the next step.

The positive class is:

```text
reasoning_needed = true
```

False-fast errors are safety-critical:

```text
actual reasoning_needed = true
predicted reasoning_needed = false
```
