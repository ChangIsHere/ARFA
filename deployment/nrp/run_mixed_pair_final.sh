#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source deployment/nrp/gpu_job_lib.sh

pair="${1:-}"
case "${pair}" in
  gemma4_qwen14)
    config="config/phase3_mixed_gemma4_qwen14.yaml"
    configmap="deployment/nrp/28-mixed-gemma4-qwen14-configmap.yaml"
    job_yaml="deployment/nrp/66-mixed-gemma4-qwen14-service.yaml"
    NRP_JOB="arfa-mixed-gemma4-qwen14-run"
    NRP_SERVICE="arfa-mixed-gemma4-qwen14"
    NRP_LOCAL_PORT="11441"
    models=(arfa-gemma3:4b-8k arfa-qwen2.5-coder:14b-8k)
    ;;
  qwen7_gemma12)
    config="config/phase3_mixed_qwen7_gemma12.yaml"
    configmap="deployment/nrp/29-mixed-qwen7-gemma12-configmap.yaml"
    job_yaml="deployment/nrp/67-mixed-qwen7-gemma12-service.yaml"
    NRP_JOB="arfa-mixed-qwen7-gemma12-run"
    NRP_SERVICE="arfa-mixed-qwen7-gemma12"
    NRP_LOCAL_PORT="11442"
    models=(arfa-qwen2.5-coder:7b-8k arfa-gemma3:12b-8k)
    ;;
  *) echo "Usage: $0 gemma4_qwen14|qwen7_gemma12" >&2; exit 2 ;;
esac

NRP_NAMESPACE="ucr-qzhang"
NRP_LOG_DIR="results/phase3_arfa/mixed_pairs/v1/logs/${pair}"
NRP_SCHEDULING_ATTEMPTS="${NRP_SCHEDULING_ATTEMPTS:-2160}"
NRP_PROVENANCE_FILES="${config} ${configmap} ${job_yaml} deployment/nrp/run_mixed_pair_final.sh deployment/nrp/gpu_job_lib.sh evidence/phase3_mixed_pairs/freeze_manifest.sha256"
result_root="results/phase3_arfa/mixed_pairs/v1"
workers=48
policies=(residual learned_observation_only random_budget always_slow always_fast)
mkdir -p "${NRP_LOG_DIR}"
LC_ALL=C LANG=C shasum -a 256 -c evidence/phase3_mixed_pairs/freeze_manifest.sha256

nrp_acquire_runner_lock
heartbeat_pid=""
cleanup_pair() {
  if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" 2>/dev/null || true
    wait "${heartbeat_pid}" 2>/dev/null || true
  fi
  nrp_cleanup
}
trap cleanup_pair EXIT
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

for policy in "${policies[@]}"; do
  policy_dir="${result_root}/final/${pair}/${policy}"
  policy_log_dir="${NRP_LOG_DIR}/${policy}"
  utilization="${policy_log_dir}/gpu-utilization.csv"
  mkdir -p "${policy_log_dir}"
  nrp_start_gpu_monitor "${utilization}"
  .venv/bin/python -m src.phase3_arfa.run_arfa \
    --config "${config}" --pair "${pair}" --policy "${policy}" \
    --split final --workers "${workers}" --output-dir "${policy_dir}" --resume \
    > "${policy_log_dir}/run.log" 2>&1
  nrp_stop_gpu_monitor
  nrp_report_gpu_utilization "${utilization}"
  jq -e '.task_count == 100 and .expected_task_count == 100 and .model_request_error_count == 0' \
    "${policy_dir}/summary.json" >/dev/null
  .venv/bin/python -m src.phase3_arfa.audit_policy_run \
    --config "${config}" --split final --pair "${pair}" --policy "${policy}" \
    --result-dir "${policy_dir}"
  nrp_verify_residency_after_run "${models[@]}"
done

.venv/bin/python -m src.phase3_arfa.analyze_results \
  --result-root "${result_root}" --split final --pair "${pair}" \
  --strict-controls --output-dir "${result_root}/analysis/final/${pair}"
