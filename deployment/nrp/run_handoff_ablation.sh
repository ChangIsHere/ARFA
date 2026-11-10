#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source deployment/nrp/gpu_job_lib.sh

config="config/phase3_handoff_qwen7_gemma12.yaml"
configmap="deployment/nrp/29-mixed-qwen7-gemma12-configmap.yaml"
job_yaml="deployment/nrp/67-mixed-qwen7-gemma12-service.yaml"
result_root="results/phase3_arfa/handoff/v1"
NRP_NAMESPACE="ucr-qzhang"
NRP_JOB="arfa-mixed-qwen7-gemma12-run"
NRP_SERVICE="arfa-mixed-qwen7-gemma12"
NRP_LOCAL_PORT="11442"
NRP_LOG_DIR="${result_root}/logs"
NRP_SCHEDULING_ATTEMPTS="${NRP_SCHEDULING_ATTEMPTS:-2160}"
NRP_PROVENANCE_FILES="${config} ${configmap} ${job_yaml} deployment/nrp/run_handoff_ablation.sh deployment/nrp/gpu_job_lib.sh evidence/phase3_handoff/freeze_manifest.sha256"
models=(arfa-qwen2.5-coder:7b-8k arfa-gemma3:12b-8k)
modes=(full_visible structured_state observation_only)
mkdir -p "${NRP_LOG_DIR}"
LC_ALL=C LANG=C shasum -a 256 -c evidence/phase3_handoff/freeze_manifest.sha256

nrp_acquire_runner_lock
heartbeat_pid=""
cleanup_handoff() {
  if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" 2>/dev/null || true
    wait "${heartbeat_pid}" 2>/dev/null || true
  fi
  nrp_cleanup
}
trap cleanup_handoff EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

kubectl --context nautilus -n "${NRP_NAMESPACE}" apply -f "${configmap}"
nrp_start_job "${job_yaml}"
export OLLAMA_HOST="http://127.0.0.1:${NRP_LOCAL_PORT}"
(
  while true; do
    kubectl --context nautilus -n "${NRP_NAMESPACE}" exec "job/${NRP_JOB}" -- \
      touch /tmp/arfa-runner-heartbeat >/dev/null 2>&1 || true
    sleep 60
  done
) &
heartbeat_pid=$!

nrp_verify_runtime_contract 8 "${models[@]}"
.venv/bin/python -m src.phase3_arfa.runtime_preflight \
  --config "${config}" --output "${NRP_LOG_DIR}/runtime-preflight.json"
nrp_write_runtime_provenance

for mode in "${modes[@]}"; do
  run_dir="${result_root}/development/${mode}/residual"
  log_dir="${NRP_LOG_DIR}/${mode}"
  utilization="${log_dir}/gpu-utilization.csv"
  mkdir -p "${log_dir}"
  nrp_start_gpu_monitor "${utilization}"
  .venv/bin/python -m src.phase3_arfa.run_arfa \
    --config "${config}" --pair "${mode}" --policy residual \
    --split development --workers 48 --output-dir "${run_dir}" --resume \
    > "${log_dir}/run.log" 2>&1
  nrp_stop_gpu_monitor
  nrp_report_gpu_utilization "${utilization}"
  jq -e '.task_count == 100 and .expected_task_count == 100 and .model_request_error_count == 0' \
    "${run_dir}/summary.json" >/dev/null
  .venv/bin/python -m src.phase3_arfa.audit_policy_run \
    --config "${config}" --split development --pair "${mode}" --policy residual \
    --result-dir "${run_dir}"
  nrp_verify_residency_after_run "${models[@]}"
done

.venv/bin/python -m src.phase3_arfa.analyze_handoff \
  --result-root "${result_root}" --output-dir "${result_root}/analysis"
