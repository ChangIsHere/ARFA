#!/usr/bin/env bash

# Shared lifecycle helpers for bounded NRP inference Jobs.
NRP_PORT_FORWARD_PID=""
NRP_GPU_MONITOR_PID=""
NRP_JOB_OWNED=""
NRP_LOCK_OWNED=""
NRP_LOCK_DIR="${NRP_LOCK_DIR:-/tmp/arfa-nrp-gpu-runner.lock}"

nrp_acquire_runner_lock() {
  local owner_pid=""
  if mkdir "${NRP_LOCK_DIR}" 2>/dev/null; then
    printf '%s\n' "$$" > "${NRP_LOCK_DIR}/pid"
    NRP_LOCK_OWNED="1"
    return 0
  fi

  if [[ -f "${NRP_LOCK_DIR}/pid" ]]; then
    owner_pid=$(cat "${NRP_LOCK_DIR}/pid" 2>/dev/null || true)
  fi
  echo "ARFA NRP runner lock already exists (recorded pid=${owner_pid:-unknown})." >&2
  echo "Inspect the recorded process and remove ${NRP_LOCK_DIR} only if it is stale." >&2
  return 1
}

nrp_release_runner_lock() {
  if [[ "${NRP_LOCK_OWNED}" == "1" ]]; then
    rm -f "${NRP_LOCK_DIR}/pid"
    rmdir "${NRP_LOCK_DIR}" 2>/dev/null || true
    NRP_LOCK_OWNED=""
  fi
}

nrp_capture_startup_diagnostics() {
  local pod_name="${1:-}"
  mkdir -p "${NRP_LOG_DIR}"
  kubectl --context nautilus -n "${NRP_NAMESPACE}" get job "${NRP_JOB}" -o yaml \
    > "${NRP_LOG_DIR}/startup-job.yaml" 2>&1 || true
  if [[ -n "${pod_name}" ]]; then
    kubectl --context nautilus -n "${NRP_NAMESPACE}" get pod "${pod_name}" -o yaml \
      > "${NRP_LOG_DIR}/startup-pod.yaml" 2>&1 || true
    kubectl --context nautilus -n "${NRP_NAMESPACE}" describe pod "${pod_name}" \
      > "${NRP_LOG_DIR}/startup-pod-describe.txt" 2>&1 || true
    kubectl --context nautilus -n "${NRP_NAMESPACE}" logs "${pod_name}" \
      > "${NRP_LOG_DIR}/startup-container.log" 2>&1 || true
  fi
}

nrp_capture_exit_diagnostics() {
  local pod_name=""
  mkdir -p "${NRP_LOG_DIR}"
  pod_name=$(kubectl --context nautilus -n "${NRP_NAMESPACE}" get pod \
    -l "job-name=${NRP_JOB}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  kubectl --context nautilus -n "${NRP_NAMESPACE}" get job "${NRP_JOB}" -o yaml \
    > "${NRP_LOG_DIR}/exit-job.yaml" 2>&1 || true
  if [[ -n "${pod_name}" ]]; then
    kubectl --context nautilus -n "${NRP_NAMESPACE}" get pod "${pod_name}" -o yaml \
      > "${NRP_LOG_DIR}/exit-pod.yaml" 2>&1 || true
    kubectl --context nautilus -n "${NRP_NAMESPACE}" describe pod "${pod_name}" \
      > "${NRP_LOG_DIR}/exit-pod-describe.txt" 2>&1 || true
    kubectl --context nautilus -n "${NRP_NAMESPACE}" logs "${pod_name}" \
      > "${NRP_LOG_DIR}/exit-container.log" 2>&1 || true
  fi
}

nrp_cleanup() {
  if [[ -n "${NRP_GPU_MONITOR_PID}" ]]; then
    kill "${NRP_GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${NRP_GPU_MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${NRP_PORT_FORWARD_PID}" ]]; then
    kill "${NRP_PORT_FORWARD_PID}" 2>/dev/null || true
    wait "${NRP_PORT_FORWARD_PID}" 2>/dev/null || true
  fi
  if [[ "${NRP_JOB_OWNED}" == "1" ]]; then
    nrp_capture_exit_diagnostics
    kubectl --context nautilus -n "${NRP_NAMESPACE}" delete job "${NRP_JOB}" \
      --ignore-not-found --wait=false >/dev/null 2>&1 || true
    NRP_JOB_OWNED=""
  fi
  nrp_release_runner_lock
}

nrp_port_forward_supervisor() {
  local child=""
  trap 'if [[ -n "${child}" ]]; then kill "${child}" 2>/dev/null || true; fi; exit 0' TERM INT
  while true; do
    kubectl --context nautilus -n "${NRP_NAMESPACE}" port-forward \
      "service/${NRP_SERVICE}" "${NRP_LOCAL_PORT}:11434" \
      >> "${NRP_LOG_DIR}/port-forward.log" 2>&1 &
    child=$!
    wait "${child}" || true
    child=""
    sleep 1
  done
}

nrp_start_job() {
  local manifest="$1"
  local failed=""
  local pod_name=""
  local ready=""
  mkdir -p "${NRP_LOG_DIR}"
  rm -f "${NRP_LOG_DIR}/startup-job.yaml" \
    "${NRP_LOG_DIR}/startup-pod.yaml" \
    "${NRP_LOG_DIR}/startup-pod-describe.txt" \
    "${NRP_LOG_DIR}/startup-container.log"
  if kubectl --context nautilus -n "${NRP_NAMESPACE}" get job "${NRP_JOB}" >/dev/null 2>&1; then
    echo "Refusing to replace existing Job ${NRP_JOB}. Inspect or delete it explicitly." >&2
    return 1
  fi
  kubectl --context nautilus apply -f "${manifest}"
  NRP_JOB_OWNED="1"

  local attempts="${NRP_SCHEDULING_ATTEMPTS:-360}"
  for _ in $(seq 1 "${attempts}"); do
    pod_name=$(kubectl --context nautilus -n "${NRP_NAMESPACE}" get pod \
      -l "job-name=${NRP_JOB}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    failed=$(kubectl --context nautilus -n "${NRP_NAMESPACE}" get job "${NRP_JOB}" \
      -o jsonpath='{.status.failed}' 2>/dev/null || true)
    if [[ -n "${failed}" && "${failed}" != "0" ]]; then
      nrp_capture_startup_diagnostics "${pod_name}"
      echo "GPU Job failed before becoming ready" >&2
      return 1
    fi
    if [[ -n "${pod_name}" ]]; then
      ready=$(kubectl --context nautilus -n "${NRP_NAMESPACE}" get pod "${pod_name}" \
        -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null || true)
      if [[ "${ready}" == "true" ]]; then
        break
      fi
    fi
    sleep 10
  done
  if [[ "${ready}" != "true" ]]; then
    nrp_capture_startup_diagnostics "${pod_name}"
    echo "GPU Job did not become ready after $((attempts * 10)) seconds" >&2
    return 1
  fi

  nrp_port_forward_supervisor &
  NRP_PORT_FORWARD_PID=$!
  for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${NRP_LOCAL_PORT}/api/tags" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Inference endpoint did not become reachable" >&2
  return 1
}

nrp_verify_runtime_contract() {
  local expected_parallel="$1"
  shift
  local models=("$@")
  local expected_models="${#models[@]}"
  local runtime_env
  local resident
  local server_log
  local parallel_runner_count

  runtime_env=$(kubectl --context nautilus -n "${NRP_NAMESPACE}" exec "job/${NRP_JOB}" -- \
    /bin/bash -lc 'printf "max_loaded=%s\nparallel=%s\nkeep_alive=%s\n" \
      "${OLLAMA_MAX_LOADED_MODELS}" "${OLLAMA_NUM_PARALLEL}" "${OLLAMA_KEEP_ALIVE}"')
  printf '%s\n' "${runtime_env}" | tee "${NRP_LOG_DIR}/ollama-runtime-contract.txt"
  grep -qx "max_loaded=${expected_models}" <<< "${runtime_env}"
  grep -qx "parallel=${expected_parallel}" <<< "${runtime_env}"
  grep -qx 'keep_alive=-1' <<< "${runtime_env}"

  resident=$(kubectl --context nautilus -n "${NRP_NAMESPACE}" exec "job/${NRP_JOB}" -- ollama ps)
  printf '%s\n' "${resident}" | tee "${NRP_LOG_DIR}/ollama-ps-before-smoke.txt"
  for model in "${models[@]}"; do
    local model_line
    model_line=$(grep -F "${model}" <<< "${resident}")
    grep -q '100% GPU' <<< "${model_line}"
    grep -q 'Forever' <<< "${model_line}"
  done

  kubectl --context nautilus -n "${NRP_NAMESPACE}" exec "job/${NRP_JOB}" -- \
    nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,driver_version \
      --format=csv,noheader | tee "${NRP_LOG_DIR}/nvidia-smi-before-smoke.csv"
  server_log=$(kubectl --context nautilus -n "${NRP_NAMESPACE}" exec "job/${NRP_JOB}" -- \
    /bin/bash -lc 'cat "/workspace/runs/${ARFA_RUN_ID}/environment/ollama-server.log"')
  printf '%s\n' "${server_log}" > "${NRP_LOG_DIR}/ollama-server-before-smoke.log"
  parallel_runner_count=$(grep -Ec "n_seq_max[[:space:]]*=[[:space:]]*${expected_parallel}$" \
    <<< "${server_log}" || true)
  if [[ "${parallel_runner_count}" -lt "${expected_models}" ]]; then
    echo "Expected ${expected_models} resident runners with n_seq_max=${expected_parallel}; found ${parallel_runner_count}." >&2
    return 1
  fi

  nrp_write_runtime_provenance
}

nrp_write_runtime_provenance() {
  local pod_name
  local dirty="false"
  local path

  pod_name=$(kubectl --context nautilus -n "${NRP_NAMESPACE}" get pod \
    -l "job-name=${NRP_JOB}" -o jsonpath='{.items[0].metadata.name}')
  if ! git diff --quiet || ! git diff --cached --quiet; then
    dirty="true"
  fi

  {
    printf 'captured_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'git_worktree_dirty=%s\n' "${dirty}"
    printf 'job=%s\n' "${NRP_JOB}"
    printf 'pod=%s\n' "${pod_name}"
    kubectl --context nautilus -n "${NRP_NAMESPACE}" get pod "${pod_name}" \
      -o jsonpath='node={.spec.nodeName}{"\n"}pod_uid={.metadata.uid}{"\n"}image={.status.containerStatuses[0].image}{"\n"}image_id={.status.containerStatuses[0].imageID}{"\n"}'
    .venv/bin/python --version
    .venv/bin/python -m pip show PyYAML requests 2>/dev/null \
      | grep -E '^(Name|Version):' || true
    printf '%s\n' 'sha256:'
    for path in ${NRP_PROVENANCE_FILES:-}; do
      sha256sum "${path}"
    done
  } | tee "${NRP_LOG_DIR}/runtime-provenance.txt"
}

nrp_verify_residency_after_run() {
  local models=("$@")
  local resident

  resident=$(kubectl --context nautilus -n "${NRP_NAMESPACE}" exec "job/${NRP_JOB}" -- ollama ps)
  printf '%s\n' "${resident}" | tee "${NRP_LOG_DIR}/ollama-ps-after-run.txt"
  for model in "${models[@]}"; do
    local model_line
    model_line=$(grep -F "${model}" <<< "${resident}")
    grep -q '100% GPU' <<< "${model_line}"
    grep -q 'Forever' <<< "${model_line}"
  done
}

nrp_start_gpu_monitor() {
  local output="$1"
  printf 'unix_time,gpu_index,gpu_utilization_percent,memory_used_mib,memory_total_mib,power_draw_watts\n' > "${output}"
  (
    while kubectl --context nautilus -n "${NRP_NAMESPACE}" get job "${NRP_JOB}" >/dev/null 2>&1; do
      if values=$(kubectl --context nautilus -n "${NRP_NAMESPACE}" exec "job/${NRP_JOB}" -- \
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits 2>/dev/null); then
        sample_time=$(date +%s)
        while IFS= read -r gpu_values; do
          printf '%s,%s\n' "${sample_time}" "${gpu_values}" >> "${output}"
        done <<< "${values}"
      fi
      sleep 5
    done
  ) &
  NRP_GPU_MONITOR_PID=$!
}

nrp_stop_gpu_monitor() {
  if [[ -n "${NRP_GPU_MONITOR_PID}" ]]; then
    kill "${NRP_GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${NRP_GPU_MONITOR_PID}" 2>/dev/null || true
    NRP_GPU_MONITOR_PID=""
  fi
}

nrp_average_gpu_utilization() {
  local input="$1"
  awk -F, '
    function flush_sample() {
      if (sample_count > 0) {
        total += sample_total / sample_count
        samples += 1
      }
    }
    NR > 1 {
      for (i = 1; i <= NF; i++) gsub(/[[:space:]]/, "", $i)
      if ($1 ~ /^[0-9]+$/ && $3 ~ /^[0-9.]+$/) {
        if (have_sample && $1 != sample_time) {
          flush_sample()
          sample_total = 0
          sample_count = 0
        }
        sample_time = $1
        sample_total += $3
        sample_count += 1
        have_sample = 1
      }
    }
    END {
      flush_sample()
      if (samples == 0) exit 2
      printf "%.2f\n", total / samples
    }
  ' "${input}"
}

nrp_gpu_energy_summary() {
  local input="$1"
  awk -F, '
    function flush_sample() {
      if (!have_sample) return
      if (have_previous && sample_time > previous_time) {
        elapsed = sample_time - previous_time
        joules += elapsed * (previous_power + sample_power) / 2
        duration += elapsed
      }
      previous_time = sample_time
      previous_power = sample_power
      power_total += sample_power
      power_count += 1
      have_previous = 1
    }
    NR > 1 {
      for (i = 1; i <= NF; i++) gsub(/[[:space:]]/, "", $i)
      if ($1 ~ /^[0-9]+$/ && $6 ~ /^[0-9.]+$/) {
        if (have_sample && $1 != sample_time) {
          flush_sample()
          sample_power = 0
        }
        sample_time = $1
        sample_power += $6
        have_sample = 1
      }
    }
    END {
      flush_sample()
      if (power_count == 0) {
        print "average_sampled_power_watts=unavailable"
        print "sampled_duration_seconds=0"
        print "estimated_gpu_energy_wh=unavailable"
        exit
      }
      printf "average_sampled_power_watts=%.2f\n", power_total / power_count
      printf "sampled_duration_seconds=%.0f\n", duration
      printf "estimated_gpu_energy_wh=%.4f\n", joules / 3600
    }
  ' "${input}"
}

nrp_require_gpu_utilization() {
  local input="$1"
  local minimum="$2"
  local average
  average=$(nrp_average_gpu_utilization "${input}")
  {
    printf 'average_gpu_utilization_percent=%s\n' "${average}"
    nrp_gpu_energy_summary "${input}"
  } | tee "${input%.csv}-summary.txt"
  awk -v observed="${average}" -v required="${minimum}" \
    'BEGIN { exit !(observed + 0 >= required + 0) }'
}

nrp_report_gpu_utilization() {
  local input="$1"
  local average
  average=$(nrp_average_gpu_utilization "${input}")
  {
    printf 'average_gpu_utilization_percent=%s\n' "${average}"
    nrp_gpu_energy_summary "${input}"
  } | tee "${input%.csv}-summary.txt"
}
