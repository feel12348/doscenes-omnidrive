#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-projects/configs/OmniDrive/mask_eva_lane_det_vlm_doscenes_only_test_new.py}"
CHECKPOINT="${CHECKPOINT:-}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29511}"
DCSV="${DCSV:-data/annotated_doscenes_test.csv}"
SAVE_DIR="${SAVE_DIR:-results_doscenes_new_official150}"
SMOKE_MAX_SCENES="${SMOKE_MAX_SCENES:-0}"

required_files=(
  "${CONFIG}"
  "${DCSV}"
  "tools/test_doscenes_challenge_submit.py"
  "data/nuscenes/nuscenes2d_ego_temporal_infos_test.pkl"
)

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[doScenesTest][ERROR] CHECKPOINT is required, e.g. CHECKPOINT=/path/to/latest.pth" >&2
  exit 1
fi

required_files+=("${CHECKPOINT}")

for path in "${required_files[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[doScenesTest][ERROR] missing required file: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${SAVE_DIR}"

echo "[doScenesTest] CONFIG=${CONFIG}"
echo "[doScenesTest] CHECKPOINT=${CHECKPOINT}"
echo "[doScenesTest] GPUS=${GPUS}"
echo "[doScenesTest] SAVE_DIR=${SAVE_DIR}"
echo "[doScenesTest] DCSV=${DCSV}"

common_args=(
  "tools/test_doscenes_challenge_submit.py"
  "${CONFIG}"
  "${CHECKPOINT}"
  "--doscenes-csv" "${DCSV}"
  "--save-dir" "${SAVE_DIR}"
  "--submission-csv" "${SAVE_DIR}/submission.csv"
  "--records-json" "${SAVE_DIR}/prediction_records.json"
  "--official-150"
)

if [[ "${SMOKE_MAX_SCENES}" != "0" ]]; then
  common_args+=("--max-scenes" "${SMOKE_MAX_SCENES}")
fi

if [[ "${GPUS}" == "1" ]]; then
  python "${common_args[@]}"
else
  torchrun --nproc_per_node="${GPUS}" --master_port="${MASTER_PORT}" \
    "${common_args[@]}" \
    --launcher pytorch
fi

echo "[doScenesTest] submission: ${SAVE_DIR}/submission.csv"
