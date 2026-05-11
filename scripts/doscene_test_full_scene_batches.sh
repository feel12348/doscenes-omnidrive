#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXP_NAME="${EXP_NAME:-exp03}"
CONDA_ENV="${CONDA_ENV:-omnidrive}"
GPU_SET_A="${GPU_SET_A:-0,1,2,3}"
GPU_SET_B="${GPU_SET_B:-4,5,6,7}"
PORT_A="${PORT_A:-29501}"
PORT_B="${PORT_B:-29502}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/log_test/runner_logs/${EXP_NAME}}"
DRY_RUN="${DRY_RUN:-0}"

preflight_check() {
  local bad=0

  if [[ ! -f "${REPO_ROOT}/scripts/doscene_test_full_scene.sh" ]]; then
    echo "Missing script: scripts/doscene_test_full_scene.sh"
    bad=1
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found in PATH"
    bad=1
  fi

  if [[ "${PORT_A}" == "${PORT_B}" ]]; then
    echo "PORT_A and PORT_B must be different"
    bad=1
  fi

  if [[ -z "${GPU_SET_A}" || -z "${GPU_SET_B}" ]]; then
    echo "GPU_SET_A and GPU_SET_B must be non-empty"
    bad=1
  fi

  if [[ "${bad}" -ne 0 ]]; then
    exit 1
  fi
}

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${LOG_DIR}"
fi

run_pair() {
  local pair_name="$1"
  local cmd_a="$2"
  local cmd_b="$3"
  local log_a="${LOG_DIR}/${pair_name}_lang.log"
  local log_b="${LOG_DIR}/${pair_name}_nolang.log"

  echo "============================================================"
  echo "Starting pair: ${pair_name}"
  echo "  Task A -> GPUs ${GPU_SET_A}, MASTER_PORT=${PORT_A}"
  echo "  Task B -> GPUs ${GPU_SET_B}, MASTER_PORT=${PORT_B}"
  echo "  Logs:"
  echo "    ${log_a}"
  echo "    ${log_b}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  DRY_RUN=1, commands only:"
    echo "    ${cmd_a}"
    echo "    ${cmd_b}"
    echo "Finished pair: ${pair_name} (dry run)"
    return 0
  fi

  (
    cd "${REPO_ROOT}"
    eval "${cmd_a}"
  ) > >(tee "${log_a}") 2>&1 &
  local pid_a=$!

  (
    cd "${REPO_ROOT}"
    eval "${cmd_b}"
  ) > >(tee "${log_b}") 2>&1 &
  local pid_b=$!

  local status_a=0
  local status_b=0

  wait "${pid_a}" || status_a=$?
  wait "${pid_b}" || status_b=$?

  if [[ "${status_a}" -ne 0 || "${status_b}" -ne 0 ]]; then
    echo "Pair failed: ${pair_name}"
    echo "  Task A exit code: ${status_a}"
    echo "  Task B exit code: ${status_b}"
    exit 1
  fi

  echo "Finished pair: ${pair_name}"
}

CMD_FULL_LANG="CUDA_VISIBLE_DEVICES=${GPU_SET_A} MASTER_PORT=${PORT_A} NO_INSTRUCTION=0 EXP_NAME=${EXP_NAME} CONDA_ENV=${CONDA_ENV} bash scripts/doscene_test_full_scene.sh"
CMD_FULL_NOLANG="CUDA_VISIBLE_DEVICES=${GPU_SET_B} MASTER_PORT=${PORT_B} NO_INSTRUCTION=1 EXP_NAME=${EXP_NAME} CONDA_ENV=${CONDA_ENV} bash scripts/doscene_test_full_scene.sh"

preflight_check
run_pair "full_scene_pair" "${CMD_FULL_LANG}" "${CMD_FULL_NOLANG}"

echo "============================================================"
echo "Full-scene runs completed successfully."
echo "EXP_NAME=${EXP_NAME}"
