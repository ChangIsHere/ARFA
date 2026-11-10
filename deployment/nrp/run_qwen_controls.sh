#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source deployment/nrp/gpu_job_lib.sh

requested_policy="${1:-}"
mode="${2:-smoke}"
case "${requested_policy}" in
  residual|always_slow|always_fast|observation_only|learned_observation_only|expectation_gate_only|random_budget) ;;
  matrix) ;;
  *) echo "Usage: $0 POLICY|matrix [smoke|full]" >&2; exit 2 ;;
esac
case "${mode}" in
  smoke|full) ;;
  *) echo "Usage: $0 POLICY [smoke|full]" >&2; exit 2 ;;
esac

NRP_NAMESPACE="ucr-qzhang"
NRP_JOB="arfa-qwen-controls-run"
NRP_SERVICE="arfa-qwen-controls"
NRP_LOCAL_PORT="11438"
log_root="results/phase3_arfa/nrp/nrp-qwen-controls-v5/logs"
NRP_LOG_DIR="${log_root}/${requested_policy}"
NRP_PROVENANCE_FILES="config/phase3_qwen_controls.yaml deployment/nrp/25-qwen-controls-configmap.yaml deployment/nrp/60-qwen-controls-service.yaml deployment/nrp/run_qwen_controls.sh deployment/nrp/gpu_job_lib.sh"
config="config/phase3_qwen_controls.yaml"
result_root="results/phase3_arfa/nrp/nrp-qwen-controls-v5"
workers=48
inference_parallel=4
mkdir -p "${NRP_LOG_DIR}"

if [[ "${requested_policy}" == "matrix" ]]; then
  policies=(residual learned_observation_only expectation_gate_only random_budget)
else
  policies=("${requested_policy}")
fi

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

kubectl --context nautilus -n "${NRP_NAMESPACE}" apply -f deployment/nrp/25-qwen-controls-configmap.yaml
if [[ "${NRP_REUSE_EXISTING_JOB:-0}" == "1" ]]; then
  kubectl --context nautilus -n "${NRP_NAMESPACE}" get job "${NRP_JOB}" >/dev/null
  NRP_JOB_OWNED="1"
  nrp_port_forward_supervisor &
  NRP_PORT_FORWARD_PID=$!
  for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${NRP_LOCAL_PORT}/api/tags" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  curl -fsS "http://127.0.0.1:${NRP_LOCAL_PORT}/api/tags" >/dev/null
else
  nrp_start_job deployment/nrp/60-qwen-controls-service.yaml
fi
export OLLAMA_HOST="http://127.0.0.1:${NRP_LOCAL_PORT}"
(
  while true; do
    kubectl --context nautilus -n "${NRP_NAMESPACE}" exec "job/${NRP_JOB}" -- \
      touch /tmp/arfa-runner-heartbeat >/dev/null 2>&1 || true
    sleep 60
  done
) &
heartbeat_pid=$!
nrp_verify_runtime_contract "${inference_parallel}" \
  'arfa-qwen2.5-coder:7b-8k' 'arfa-qwen2.5-coder:14b-8k'

.venv/bin/python -m src.phase3_arfa.runtime_preflight \
  --config "${config}" --output "${NRP_LOG_DIR}/runtime-preflight.json"

run_policy() {
  local policy="$1"
  if [[ "$2" == "smoke" ]]; then
    .venv/bin/python -m src.phase3_arfa.run_arfa \
      --config "${config}" --pair qwen --policy "${policy}" \
      --split development --workers "${workers}" --task-limit 32 \
      --output-dir "${result_root}/smoke/${policy}" --resume \
      > "${log_root}/${policy}/$2-run.log" 2>&1
  else
    .venv/bin/python -m src.phase3_arfa.run_arfa \
      --config "${config}" --pair qwen --policy "${policy}" \
      --split development --workers "${workers}" --resume \
      > "${log_root}/${policy}/$2-run.log" 2>&1
  fi
}

if [[ "${requested_policy}" == "matrix" ]]; then
  matrix_log_dir="${log_root}/matrix"
  mkdir -p "${matrix_log_dir}"
  for policy in "${policies[@]}"; do mkdir -p "${log_root}/${policy}"; done
  if [[ "${ARFA_SKIP_SMOKE:-0}" != "1" ]]; then
    smoke_metrics="${matrix_log_dir}/gpu-utilization-smoke.csv"
    nrp_start_gpu_monitor "${smoke_metrics}"
    pids=()
    for policy in "${policies[@]}"; do
      run_policy "${policy}" smoke &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "${pid}"; done
    nrp_stop_gpu_monitor
    nrp_require_gpu_utilization "${smoke_metrics}" 40
    for policy in "${policies[@]}"; do
      jq -e '.task_count == 32 and .model_request_error_count == 0' \
        "${result_root}/smoke/${policy}/summary.json" >/dev/null
    done
    nrp_verify_residency_after_run 'arfa-qwen2.5-coder:7b-8k' 'arfa-qwen2.5-coder:14b-8k'
  fi

  if [[ "${mode}" == "full" ]]; then
    for policy in "${policies[@]}"; do
      full_metrics="${log_root}/${policy}/gpu-utilization-full.csv"
      nrp_start_gpu_monitor "${full_metrics}"
      run_policy "${policy}" full
      nrp_stop_gpu_monitor
      nrp_require_gpu_utilization "${full_metrics}" 40
      full_dir="${result_root}/development/qwen/${policy}"
      jq -e '.task_count == 100 and .expected_task_count == 100 and .model_request_error_count == 0' \
        "${full_dir}/summary.json" >/dev/null
      .venv/bin/python -m src.phase3_arfa.audit_policy_run \
        --config "${config}" --split development --pair qwen --policy "${policy}" \
        --result-dir "${full_dir}"
      nrp_verify_residency_after_run 'arfa-qwen2.5-coder:7b-8k' 'arfa-qwen2.5-coder:14b-8k'
    done
    .venv/bin/python -m src.phase3_arfa.analyze_results \
      --result-root "${result_root}" --split development --pair qwen --strict-controls
  fi
  exit 0
fi

for policy in "${policies[@]}"; do
  NRP_LOG_DIR="${log_root}/${policy}"
  mkdir -p "${NRP_LOG_DIR}"
  nrp_write_runtime_provenance

  smoke_metrics="${NRP_LOG_DIR}/gpu-utilization-smoke.csv"
  nrp_start_gpu_monitor "${smoke_metrics}"
  .venv/bin/python -m src.phase3_arfa.run_arfa \
    --config "${config}" --pair qwen --policy "${policy}" \
    --split development --workers "${workers}" --task-limit 32 \
    --output-dir "${result_root}/smoke/${policy}" --resume
  nrp_stop_gpu_monitor
  nrp_require_gpu_utilization "${smoke_metrics}" 40
  jq -e '.task_count == 32 and .model_request_error_count == 0' \
    "${result_root}/smoke/${policy}/summary.json" >/dev/null
  nrp_verify_residency_after_run 'arfa-qwen2.5-coder:7b-8k' 'arfa-qwen2.5-coder:14b-8k'

  if [[ "${mode}" == "full" ]]; then
    full_metrics="${NRP_LOG_DIR}/gpu-utilization-full.csv"
    nrp_start_gpu_monitor "${full_metrics}"
    .venv/bin/python -m src.phase3_arfa.run_arfa \
      --config "${config}" --pair qwen --policy "${policy}" \
      --split development --workers "${workers}" --resume
    nrp_stop_gpu_monitor
    nrp_require_gpu_utilization "${full_metrics}" 40
    full_dir="${result_root}/development/qwen/${policy}"
    jq -e '.task_count == 100 and .expected_task_count == 100 and .model_request_error_count == 0' \
      "${full_dir}/summary.json" >/dev/null
    .venv/bin/python -m src.phase3_arfa.audit_policy_run \
      --config "${config}" --split development --pair qwen --policy "${policy}" \
      --result-dir "${full_dir}"
    nrp_verify_residency_after_run 'arfa-qwen2.5-coder:7b-8k' 'arfa-qwen2.5-coder:14b-8k'
  fi
done
