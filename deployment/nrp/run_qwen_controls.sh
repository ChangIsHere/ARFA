#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source deployment/nrp/gpu_job_lib.sh

policy="${1:-}"
mode="${2:-smoke}"
case "${policy}" in
  residual|always_slow|always_fast|observation_only|learned_observation_only|expectation_gate_only|random_budget|fixed_budget) ;;
  *) echo "Usage: $0 POLICY [smoke|full]" >&2; exit 2 ;;
esac
case "${mode}" in
  smoke|full) ;;
  *) echo "Usage: $0 POLICY [smoke|full]" >&2; exit 2 ;;
esac

NRP_NAMESPACE="ucr-qzhang"
NRP_JOB="arfa-qwen-controls-run"
NRP_SERVICE="arfa-qwen-controls"
NRP_LOCAL_PORT="11435"
NRP_LOG_DIR="results/phase3_arfa/nrp/nrp-qwen-controls-v1/logs/${policy}"
NRP_PROVENANCE_FILES="config/phase3_qwen_controls.yaml deployment/nrp/60-qwen-controls-service.yaml deployment/nrp/run_qwen_controls.sh deployment/nrp/gpu_job_lib.sh"
config="config/phase3_qwen_controls.yaml"
result_root="results/phase3_arfa/nrp/nrp-qwen-controls-v1"
mkdir -p "${NRP_LOG_DIR}"

nrp_acquire_runner_lock
heartbeat_pid=""
cleanup_controls() {
  if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" 2>/dev/null || true
    wait "${heartbeat_pid}" 2>/dev/null || true
  fi
  nrp_cleanup
}
trap cleanup_controls EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

kubectl --context nautilus -n "${NRP_NAMESPACE}" apply -f deployment/nrp/20-qwen-configmap.yaml
nrp_start_job deployment/nrp/60-qwen-controls-service.yaml
(
  while true; do
    kubectl --context nautilus -n "${NRP_NAMESPACE}" exec "job/${NRP_JOB}" -- \
      touch /tmp/arfa-runner-heartbeat >/dev/null 2>&1 || true
    sleep 60
  done
) &
heartbeat_pid=$!
nrp_verify_runtime_contract 8 'arfa-qwen2.5-coder:7b-8k' 'arfa-qwen2.5-coder:14b-8k'

.venv/bin/python -m src.phase3_arfa.runtime_preflight \
  --config "${config}" --output "${NRP_LOG_DIR}/runtime-preflight.json"

smoke_metrics="${NRP_LOG_DIR}/gpu-utilization-smoke.csv"
nrp_start_gpu_monitor "${smoke_metrics}"
.venv/bin/python -m src.phase3_arfa.run_arfa \
  --config "${config}" --pair qwen --policy "${policy}" \
  --split development --workers 8 --task-limit 12 \
  --output-dir "${result_root}/smoke/${policy}" --resume
nrp_stop_gpu_monitor
nrp_report_gpu_utilization "${smoke_metrics}"
jq -e '.task_count == 12 and .model_request_error_count == 0' \
  "${result_root}/smoke/${policy}/summary.json" >/dev/null
nrp_verify_residency_after_run 'arfa-qwen2.5-coder:7b-8k' 'arfa-qwen2.5-coder:14b-8k'

if [[ "${mode}" == "full" ]]; then
  full_metrics="${NRP_LOG_DIR}/gpu-utilization-full.csv"
  nrp_start_gpu_monitor "${full_metrics}"
  .venv/bin/python -m src.phase3_arfa.run_arfa \
    --config "${config}" --pair qwen --policy "${policy}" \
    --split development --workers 8 --resume
  nrp_stop_gpu_monitor
  nrp_report_gpu_utilization "${full_metrics}"
  nrp_verify_residency_after_run 'arfa-qwen2.5-coder:7b-8k' 'arfa-qwen2.5-coder:14b-8k'
fi
