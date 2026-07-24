# Introduction

Terminal-based coding agents operate through a loop: reason about the task, run a command, observe terminal output, and decide what to do next. Many systems invoke the reasoning model after every observation. This is safe but potentially inefficient, especially when the observation simply confirms the current plan.

ARFA asks whether the agent can use execution-level feedback to decide when another reasoning step is necessary. The key signal is residual: whether the actual terminal observation deviates from what the agent expected before execution.

The first phase of this project does not build the full ARFA agent. It validates whether residual is predictive enough to justify later routing experiments.
