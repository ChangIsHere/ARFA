#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source deployment/nrp/gpu_job_lib.sh

NRP_NAMESPACE="ucr-qzhang"
NRP_JOB="arfa-gemma-high-memory-inference-run"
NRP_SERVICE="arfa-gemma-high-memory-inference"
NRP_LOCAL_PORT="11437"
result_root="results/phase3_arfa/nrp/nrp-gemma-high-memory-parallel4-workers16-development"
config="config/phase3_arfa_nrp_gemma_high_memory.yaml"
thresholds=(0.15 0.20 0.25 0.30)
workers=16
NRP_SCHEDULING_ATTEMPTS="${NRP_SCHEDULING_ATTEMPTS:-360}"
NRP_PROVENANCE_FILES="${config} deployment/nrp/22-gemma-configmap.yaml deployment/nrp/50-gemma-high-memory-inference-service.yaml deployment/nrp/run_gemma_high_memory_matrix.sh deployment/nrp/gpu_job_lib.sh"
mode="${1:-full}"
if [[ "${mode}" != "smoke" && "${mode}" != "full" && "${mode}" != "full-only" ]]; then
  echo "Usage: $0 [smoke|full|full-only]" >&2
  exit 2
fi

NRP_LOG_DIR="${result_root}/logs"
mkdir -p "${NRP_LOG_DIR}"
nrp_acquire_runner_lock
trap nrp_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

kubectl --context nautilus -n "${NRP_NAMESPACE}" apply -f deployment/nrp/22-gemma-configmap.yaml
nrp_start_job deployment/nrp/50-gemma-high-memory-inference-service.yaml

gpu_name=$(kubectl --context nautilus -n "${NRP_NAMESPACE}" exec "job/${NRP_JOB}" -- \
  nvidia-smi --query-gpu=name --format=csv,noheader)
case "${gpu_name}" in
  'NVIDIA A40'|'NVIDIA L40'|'NVIDIA L40S'|'NVIDIA RTX A6000'|\
  'NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition'|'Quadro RTX 8000'|\
  'NVIDIA A100-SXM4-80GB'|'NVIDIA A100 80GB PCIe'|'NVIDIA H100 80GB HBM3'|\
  'NVIDIA H200 NVL'|'NVIDIA GH200 480GB') ;;
  *) echo "Unexpected GPU: ${gpu_name}" >&2; exit 1 ;;
esac
kubectl --context nautilus -n "${NRP_NAMESPACE}" exec "job/${NRP_JOB}" -- \
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader \
  | tee "${NRP_LOG_DIR}/gpu-environment.txt"
nrp_verify_runtime_contract 4 'arfa-gemma3:4b-8k' 'arfa-gemma3:12b-8k'

if [[ "${mode}" != "full-only" ]]; then
  for policy in residual always_slow always_fast; do
    smoke_metrics="${NRP_LOG_DIR}/gpu-utilization-smoke-${policy}.csv"
    nrp_start_gpu_monitor "${smoke_metrics}"
    .venv/bin/python -m src.phase3_arfa.run_arfa \
      --config "${config}" \
      --pair gemma \
      --policy "${policy}" \
      --split development \
      --task-limit 12 \
      --workers "${workers}" \
      --output-dir "${result_root}/utilization_smoke/${policy}"
    smoke_summary="${result_root}/utilization_smoke/${policy}/summary.json"
    if ! jq -e \
      '.task_count == 12 and .model_request_error_count == 0 and .parse_error_count == 0' \
      "${smoke_summary}" >/dev/null; then
      echo "Smoke validation failed for ${policy}: ${smoke_summary}" >&2
      exit 1
    fi
    nrp_stop_gpu_monitor
    nrp_report_gpu_utilization "${smoke_metrics}"
    nrp_verify_residency_after_run 'arfa-gemma3:4b-8k' 'arfa-gemma3:12b-8k'
  done
fi
if [[ "${mode}" == "smoke" ]]; then
  exit 0
fi

for policy in residual always_slow always_fast; do
  full_metrics="${NRP_LOG_DIR}/gpu-utilization-full-${policy}.csv"
  nrp_start_gpu_monitor "${full_metrics}"
  .venv/bin/python -m src.phase3_arfa.run_arfa \
    --config "${config}" \
    --pair gemma \
    --policy "${policy}" \
    --split development \
    --workers "${workers}" \
    --resume
  nrp_stop_gpu_monitor
  nrp_report_gpu_utilization "${full_metrics}"
  nrp_verify_residency_after_run 'arfa-gemma3:4b-8k' 'arfa-gemma3:12b-8k'
done

calibration_metrics="${NRP_LOG_DIR}/gpu-utilization-calibration.csv"
nrp_start_gpu_monitor "${calibration_metrics}"
for threshold in "${thresholds[@]}"; do
  slug="threshold_${threshold/./_}"
  output_dir="${result_root}/calibration/${slug}/development/gemma/residual"
  .venv/bin/python -m src.phase3_arfa.run_arfa \
    --config "${config}" \
    --pair gemma \
    --policy residual \
    --split development \
    --threshold "${threshold}" \
    --workers "${workers}" \
    --output-dir "${output_dir}" \
    --resume
done
nrp_stop_gpu_monitor
nrp_report_gpu_utilization "${calibration_metrics}"

.venv/bin/python -m src.phase3_arfa.calibrate_threshold \
  --result-root "${result_root}" \
  --pair gemma \
  --split development \
  --thresholds 0.15 0.20 0.25 0.30 0.35 \
  --output-dir evidence/phase3_design/gemma_high_memory_parallel4_calibration
nrp_verify_residency_after_run 'arfa-gemma3:4b-8k' 'arfa-gemma3:12b-8k'
