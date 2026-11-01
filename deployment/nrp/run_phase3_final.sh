#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source deployment/nrp/gpu_job_lib.sh

pair="${1:-}"
case "${pair}" in
  qwen)
    config="config/phase3_final_qwen.yaml"
    manifest="evidence/phase3_final/qwen_freeze_manifest.json"
    result_root="results/phase3_arfa/final/qwen-v4"
    job_yaml="deployment/nrp/63-qwen-final-service.yaml"
    configmap_yaml="deployment/nrp/25-qwen-controls-configmap.yaml"
    NRP_JOB="arfa-qwen-final-run"
    NRP_SERVICE="arfa-qwen-final"
    NRP_LOCAL_PORT="11439"
    workers=48
    inference_parallel=8
    fast_model="arfa-qwen2.5-coder:7b-8k"
    slow_model="arfa-qwen2.5-coder:14b-8k"
    ;;
  gemma)
    config="config/phase3_final_gemma.yaml"
    manifest="evidence/phase3_final/gemma_freeze_manifest.json"
    result_root="results/phase3_arfa/final/gemma-v4"
    job_yaml="deployment/nrp/64-gemma-final-service.yaml"
    configmap_yaml="deployment/nrp/24-gemma-controls-configmap.yaml"
    NRP_JOB="arfa-gemma-final-run"
    NRP_SERVICE="arfa-gemma-final"
    NRP_LOCAL_PORT="11440"
    workers=48
    inference_parallel=8
    fast_model="arfa-gemma3:4b-8k"
    slow_model="arfa-gemma3:12b-8k"
    ;;
  *) echo "Usage: $0 qwen|gemma" >&2; exit 2 ;;
esac

policies=(residual learned_observation_only expectation_gate_only random_budget always_slow always_fast)
NRP_NAMESPACE="ucr-qzhang"
NRP_LOG_DIR="${result_root}/logs"
NRP_PROVENANCE_FILES="${config} ${manifest} ${configmap_yaml} ${job_yaml} deployment/nrp/run_phase3_final.sh deployment/nrp/gpu_job_lib.sh"
mkdir -p "${NRP_LOG_DIR}"

.venv/bin/python -m src.phase3_arfa.freeze_final \
  --pair "${pair}" --output "${manifest}" --verify

nrp_acquire_runner_lock
heartbeat_pid=""
cleanup_final() {
  if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" 2>/dev/null || true
    wait "${heartbeat_pid}" 2>/dev/null || true
  fi
  nrp_cleanup
}
trap cleanup_final EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

kubectl --context nautilus -n "${NRP_NAMESPACE}" apply -f "${configmap_yaml}"
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

nrp_verify_runtime_contract "${inference_parallel}" "${fast_model}" "${slow_model}"
.venv/bin/python -m src.phase3_arfa.runtime_preflight \
  --config "${config}" --output "${NRP_LOG_DIR}/runtime-preflight.json"
nrp_write_runtime_provenance

for policy in "${policies[@]}"; do
  policy_dir="${result_root}/final/${pair}/${policy}"
  policy_log_dir="${NRP_LOG_DIR}/${policy}"
  mkdir -p "${policy_log_dir}"
  utilization="${policy_log_dir}/gpu-utilization.csv"
  nrp_start_gpu_monitor "${utilization}"
  .venv/bin/python -m src.phase3_arfa.run_arfa \
    --config "${config}" --pair "${pair}" --policy "${policy}" \
    --split final --workers "${workers}" --output-dir "${policy_dir}" --resume \
    > "${policy_log_dir}/run.log" 2>&1
  nrp_stop_gpu_monitor
  nrp_require_gpu_utilization "${utilization}" 40
  jq -e '.task_count == 100 and .expected_task_count == 100 and .model_request_error_count == 0' \
    "${policy_dir}/summary.json" >/dev/null
  .venv/bin/python -m src.phase3_arfa.audit_policy_run \
    --config "${config}" --split final --pair "${pair}" --policy "${policy}" \
    --result-dir "${policy_dir}"
  nrp_verify_residency_after_run "${fast_model}" "${slow_model}"
done

.venv/bin/python -m src.phase3_arfa.analyze_results \
  --result-root "${result_root}" --split final --pair "${pair}" \
  --strict-controls --output-dir "${result_root}/analysis/final/${pair}"
