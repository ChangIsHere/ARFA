# ARFA on NRP Nautilus

This directory contains Kubernetes resources and bounded runners used for ARFA
model inference in namespace `ucr-qzhang`. InterCode task environments run in
isolated Docker containers on the client; only model inference runs on NRP GPUs.

The evidence bundle, not the current reusable YAML, is the record of a completed
experiment. Model tags and infrastructure templates can change after a run, so
each paper-facing result stores hashes and runtime provenance.

## Resource Rules

- Never commit kubeconfig, CILogon tokens, API keys, model weights, or caches.
- Store model caches and environment records on `arfa-nrp-workspace`.
- Use Kubernetes Jobs, not long-lived Deployments, for experiments.
- Every runner installs cleanup traps and deletes its Job after completion,
  interruption, failed residency checks, or a utilization-policy failure.
- Keep both models resident before collection and verify residency after every
  policy.
- Use identical trajectory and inference concurrency across policies in a
  matched matrix.
- Record GPU utilization, memory, power, node, image digest, model IDs, config
  hash, and Git commit.
- Do not combine partial results across different nodes or runtime contracts.

## Completed Studies

- Qwen 7B/14B native development: H100 80GB, three policies x 100 tasks.
- Llama 3B/8B native development: RTX 3090 24GB, three policies x 100 tasks.
- Gemma 4B/12B native development: L40 48GB, three policies x 100 tasks.
- Phi Mini/14B native development: L40 48GB, three policies x 100 tasks.
- Qwen and Gemma routing-control development matrices.
- Frozen Qwen 7B/14B final matrix: six policies x 100 tasks.
- Frozen Gemma 4B/12B final matrix: six policies x 100 tasks.
- Gemma 4B/Qwen 14B mixed matrix: five policies x 100 tasks.
- Qwen 7B/Gemma 12B mixed matrix: five policies x 100 tasks.
- Qwen 7B/Gemma 12B handoff ablation: three modes x 100 tasks.

Compact results and provenance are in `evidence/phase3_design/`,
`evidence/phase3_final/`, `evidence/phase3_mixed_pairs/`, and
`evidence/phase3_handoff/`. Raw logs remain in the ignored `results/` tree.

## Reusable Components

- `gpu_job_lib.sh`: shared scheduling, port forwarding, monitoring, residency,
  provenance, and cleanup helpers.
- `20-` through `25-*-configmap.yaml`: model aliases and startup checks for the
  native pairs and control matrices.
- `40-`, `41-`, `44-`, and `50-*-service.yaml`: native-pair development Jobs.
- `63-qwen-final-service.yaml` and `64-gemma-final-service.yaml`: frozen final
  Jobs.
- `66-mixed-gemma4-qwen14-service.yaml` and
  `67-mixed-qwen7-gemma12-service.yaml`: cross-family and handoff Jobs.
- `run_phase3_final.sh`: frozen Qwen/Gemma matrix runner.
- `run_mixed_pair_final.sh`: one cross-family matrix.
- `run_all_mixed_pairs.sh`: both cross-family matrices in sequence.
- `run_handoff_ablation.sh`: three handoff modes.

Development runners remain for reproducibility but are not the recommended
entry point for the paper-facing final results.

## Verification Before Reuse

```bash
kubectl --context nautilus -n ucr-qzhang apply --dry-run=server -f <manifest>
bash -n deployment/nrp/*.sh
```

Check model IDs against the corresponding evidence manifest before requesting a
GPU. Ollama library tags are mutable and must not be accepted silently.
