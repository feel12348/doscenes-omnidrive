#!/usr/bin/env bash
set -euo pipefail

# ====== 基础路径 ======
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# ====== 可改参数 ======
CONFIG="${CONFIG:-projects/configs/OmniDrive/mask_eva_lane_det_vlm.py}"
GPUS="${GPUS:-8}"
WORK_DIR="${WORK_DIR:-log_train/mask_eva_lane_det_vlm_run}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

# 0/1 开关
ENABLE_DOSCENES="${ENABLE_DOSCENES:-0}"          # 默认不注入
RANDOM_DOSCENES="${RANDOM_DOSCENES:-1}"          # 开启注入后，是否随机一条指令
ONLY_DOSCENES_SAMPLES="${ONLY_DOSCENES_SAMPLES:-0}"  # 是否只保留有指令的场景

DCSV="${DCSV:-data/annotated_doscenes.csv}"

to_py_bool () {
  if [[ "$1" == "1" ]]; then echo "True"; else echo "False"; fi
}

ENABLE_PY="$(to_py_bool "${ENABLE_DOSCENES}")"
RANDOM_PY="$(to_py_bool "${RANDOM_DOSCENES}")"
ONLY_PY="$(to_py_bool "${ONLY_DOSCENES_SAMPLES}")"

if [[ "${ENABLE_DOSCENES}" == "1" && ! -f "${DCSV}" ]]; then
  echo "[ERROR] doScenes CSV not found: ${DCSV}"
  exit 1
fi

echo "CONFIG=${CONFIG}"
echo "GPUS=${GPUS}"
echo "NUM_GPUS=${NUM_GPUS:-${GPUS}}"
echo "WORK_DIR=${WORK_DIR}"
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi
echo "ENABLE_DOSCENES=${ENABLE_PY}"
echo "RANDOM_DOSCENES=${RANDOM_PY}"
echo "ONLY_DOSCENES_SAMPLES=${ONLY_PY}"
echo "DCSV=${DCSV}"

# ====== 训练 ======
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" NUM_GPUS="${NUM_GPUS:-${GPUS}}" bash tools/dist_train.sh "${CONFIG}" "${GPUS}" \
    --work-dir "${WORK_DIR}" \
    --cfg-options \
    data.train.doscenes_csv="${DCSV}" \
    data.train.enable_doscenes_instruction="${ENABLE_PY}" \
    data.train.random_doscenes_instruction="${RANDOM_PY}" \
    data.train.only_doscenes_samples="${ONLY_PY}" \
    data.val.doscenes_csv="${DCSV}" \
    data.val.enable_doscenes_instruction="${ENABLE_PY}" \
    data.val.only_doscenes_samples="${ONLY_PY}" \
    data.test.doscenes_csv="${DCSV}" \
    data.test.enable_doscenes_instruction="${ENABLE_PY}" \
    data.test.only_doscenes_samples="${ONLY_PY}"
else
  NUM_GPUS="${NUM_GPUS:-${GPUS}}" bash tools/dist_train.sh "${CONFIG}" "${GPUS}" \
    --work-dir "${WORK_DIR}" \
    --cfg-options \
    data.train.doscenes_csv="${DCSV}" \
    data.train.enable_doscenes_instruction="${ENABLE_PY}" \
    data.train.random_doscenes_instruction="${RANDOM_PY}" \
    data.train.only_doscenes_samples="${ONLY_PY}" \
    data.val.doscenes_csv="${DCSV}" \
    data.val.enable_doscenes_instruction="${ENABLE_PY}" \
    data.val.only_doscenes_samples="${ONLY_PY}" \
    data.test.doscenes_csv="${DCSV}" \
    data.test.enable_doscenes_instruction="${ENABLE_PY}" \
    data.test.only_doscenes_samples="${ONLY_PY}"
fi
