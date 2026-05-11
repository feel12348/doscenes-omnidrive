#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OMNIDRIVE_DIR="${REPO_ROOT}"

# ===== Defaults (can be overridden by env) =====
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
CONDA_ENV="${CONDA_ENV:-omnidrive}"
MASTER_PORT="${MASTER_PORT:-29500}"
FORCE_REWRITE_PREDS="${FORCE_REWRITE_PREDS:-1}"

CONFIG="${CONFIG:-projects/configs/OmniDrive/mask_eva_lane_det_vlm_doscenes_only.py}"
CKPT="${CKPT:-ckpts/iter_10548.pth}"
DOSCENES_CSV="${DOSCENES_CSV:-data/annotated_doscenes.csv}"
OBS_LEN="${OBS_LEN:-4}"
FUT_LEN="${FUT_LEN:-12}"
PLANNING_STEPS="${PLANNING_STEPS:-6}"
WINDOW_STRIDE="${WINDOW_STRIDE:-1}"
NO_INSTRUCTION="${NO_INSTRUCTION:-0}"   # 1 => add --no-instruction

if [[ -z "${EXP_ROOT:-}" ]]; then
  if [[ "${NO_INSTRUCTION}" == "1" ]]; then
    EXP_ROOT="log_test/window_sild_nolang"
  else
    EXP_ROOT="log_test/window_sild_lang"
  fi
fi
EXP_NAME="${EXP_NAME:-exp01}"
SAVE_DIR="${SAVE_DIR:-${EXP_ROOT}/${EXP_NAME}/preds/}"
INST_JSON_DIR="${INST_JSON_DIR:-${EXP_ROOT}/${EXP_NAME}/instruction_jsons}"
METRICS_JSON="${METRICS_JSON:-${EXP_ROOT}/${EXP_NAME}/metrics.json}"

mkdir -p "${OMNIDRIVE_DIR}/${SAVE_DIR}" "${OMNIDRIVE_DIR}/${INST_JSON_DIR}"

# Infer nproc from CUDA_VISIBLE_DEVICES unless overridden.
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "FORCE_REWRITE_PREDS=${FORCE_REWRITE_PREDS}"
echo "CONFIG=${CONFIG}"
echo "CKPT=${CKPT}"
echo "DOSCENES_CSV=${DOSCENES_CSV}"
echo "EXP_ROOT=${EXP_ROOT}"
echo "EXP_NAME=${EXP_NAME}"
echo "SAVE_DIR=${SAVE_DIR}"
echo "INST_JSON_DIR=${INST_JSON_DIR}"
echo "METRICS_JSON=${METRICS_JSON}"

cd "${OMNIDRIVE_DIR}"

cmd=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  python -u -m torch.distributed.run
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  tools/test_doscenes_sliding_window_eval.py
  "${CONFIG}"
  "${CKPT}"
  --launcher pytorch
  --obs-len "${OBS_LEN}"
  --fut-len "${FUT_LEN}"
  --planning-steps "${PLANNING_STEPS}"
  --window-stride "${WINDOW_STRIDE}"
  --save-dir "${SAVE_DIR}"
  --per-instruction-json-dir "${INST_JSON_DIR}"
  --metrics-json "${METRICS_JSON}"
  --cfg-options
  data.test.enable_doscenes_instruction=True
  "data.test.doscenes_csv=${DOSCENES_CSV}"
)

if [[ "${NO_INSTRUCTION}" == "1" ]]; then
  cmd+=(--no-instruction)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" FORCE_REWRITE_PREDS="${FORCE_REWRITE_PREDS}" "${cmd[@]}"


# CUDA_VISIBLE_DEVICES=0,1,2,3 EXP_NAME=exp02 bash scripts/doscene_test_window.sh
