# ARFA on NRP Nautilus

This directory contains reusable Kubernetes resources for ARFA model inference
in namespace `ucr-qzhang`. InterCode task environments remain isolated Docker
containers on the client machine; only model inference runs on NRP GPUs. The
paper-facing record of each completed run is its evidence bundle, not the
current reusable deployment template.

## Resource Rules

- Credentials, kubeconfig, tokens, model weights, and caches are not committed.
- Model caches and NRP environment records use the `arfa-nrp-workspace` PVC.
- Every inference workload is a Kubernetes Job with `restartPolicy: Never`, a
  12-hour deadline, no retry, and a cleanup trap in its runner.
- Legacy ARFA Deployments are kept at zero replicas. Current runners do not use
  them.
- A short smoke run is required before a full matrix.
- A global local lock prevents two ARFA runners from replacing each other's
  Jobs. Runners also refuse to overwrite an existing Job.
- Before the smoke run, both model tags must be resident at `100% GPU` with
  infinite keep-alive. Both Ollama runners must report the configured parallel
  sequence count.
- GPU utilization, memory, and power draw are sampled every five seconds. Each
  of `residual`, `always_slow`, and `always_fast` must average at least 40%
  utilization separately. The power samples are integrated into an energy
  estimate for each policy.
- Failure, interruption, preemption, or a utilization result below 40% deletes
  the Job. A full matrix is not started after a failed smoke run.

## Completed Development Runs

- Qwen 7B/14B: NVIDIA H100 80GB, 100 tasks x 3 policies.
- Llama 3B/8B: NVIDIA RTX 3090 24GB, 100 tasks x 3 policies.
- Gemma 4B/12B: NVIDIA L40 48GB, 100 tasks x 3 policies plus development
  threshold calibration.
- Phi Mini/14B: NVIDIA L40 48GB, 100 tasks x 3 policies plus development
  threshold calibration.

Compact results and hardware records are under `evidence/phase3_design/`. Raw
GPU logs and traces stay in the ignored `results/phase3_arfa/nrp/` directory.
Absolute latency is compared only within a matched pair because hardware and
concurrency differ across studies.

## Parallel Execution

`src/phase3_arfa/run_arfa.py` supports bounded trajectory workers. Each worker
gets a separate InterCode container, while completed task records are written by
the main thread in task order. Per-task wall time, model time, command time,
router time, traces, and checkpoints retain the existing schema.

The same worker count is used for all three policies in a matched comparison.
`analyze_results.py` and `calibrate_threshold.py` reject results with different
worker counts.

Ollama uses `OLLAMA_NUM_PARALLEL` equal to the trajectory worker count, so
concurrent requests can be batched without changing model tags, quantization,
context length, temperature, or `keep_alive=-1`.

Completed jobs used bounded trajectory workers and identical concurrency for
the three policies within each pair. A preemption or node loss invalidates that
run; the runner terminates and does not combine records across nodes.

## Primary Files

- `gpu_job_lib.sh`: bounded Job lifecycle, GPU monitoring, provenance, and
  cleanup.
- `20-` through `23-*-configmap.yaml`: frozen model aliases and startup checks.
- `40-qwen-inference-service.yaml`, `41-llama-inference-service.yaml`,
  `50-gemma-high-memory-inference-service.yaml`, and
  `44-phi-inference-service.yaml`: reusable inference Jobs and Services.
- `run_qwen_development.sh`, `run_llama_development.sh`,
  `run_gemma_high_memory_matrix.sh`, and `run_phi_high_memory_matrix.sh`: the
  pair-specific development runners.

Before reuse, run `kubectl apply --dry-run=server` on the relevant manifest and
check the model IDs against the evidence manifest. Ollama library tags are
mutable; a changed model ID must not be accepted silently.
