#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OMNIDRIVE_DIR="${REPO_ROOT}"

# ===== Defaults (can be overridden by env) =====
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
CONDA_ENV="${CONDA_ENV:-omnidrive}"
MASTER_PORT="${MASTER_PORT:-29500}"
FORCE_REWRITE_PREDS="${FORCE_REWRITE_PREDS:-1}"

CONFIG="${CONFIG:-projects/configs/OmniDrive/mask_eva_lane_det_vlm.py}"
CKPT="${CKPT:-ckpts/iter_10548.pth}"
DOSCENES_CSV="${DOSCENES_CSV:-data/annotated_doscenes.csv}"

FRAME_STRIDE="${FRAME_STRIDE:-1}"
HISTORY_SECONDS="${HISTORY_SECONDS:-2.0}"
HISTORY_FRAMES="${HISTORY_FRAMES:-0}"  # >0 will override HISTORY_SECONDS
PLANNING_STEPS="${PLANNING_STEPS:-6}"
DROP_TAIL_FRAMES="${DROP_TAIL_FRAMES:-12}"  # drop last N frames (set 0 to disable)
NO_INSTRUCTION="${NO_INSTRUCTION:-0}"  # 1 => add --no-instruction
RECORD_WARMUP_OUTPUTS="${RECORD_WARMUP_OUTPUTS:-0}"  # 1 => add --record-warmup-outputs

if [[ -z "${EXP_ROOT:-}" ]]; then
  if [[ "${NO_INSTRUCTION}" == "1" ]]; then
    EXP_ROOT="log_test/full_scene_nolang"
  else
    EXP_ROOT="log_test/full_scene_lang"
  fi
fi
EXP_NAME="${EXP_NAME:-exp01}"
SAVE_DIR="${SAVE_DIR:-${EXP_ROOT}/${EXP_NAME}/preds/}"
INDEX_JSON="${INDEX_JSON:-${EXP_ROOT}/${EXP_NAME}/index.json}"
METRICS_JSON="${METRICS_JSON:-${EXP_ROOT}/${EXP_NAME}/metrics.json}"
mkdir -p "${OMNIDRIVE_DIR}/${SAVE_DIR}"

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
echo "INDEX_JSON=${INDEX_JSON}"
echo "METRICS_JSON=${METRICS_JSON}"
echo "FRAME_STRIDE=${FRAME_STRIDE}"
echo "HISTORY_SECONDS=${HISTORY_SECONDS}"
echo "HISTORY_FRAMES=${HISTORY_FRAMES}"
echo "DROP_TAIL_FRAMES=${DROP_TAIL_FRAMES}"

cd "${OMNIDRIVE_DIR}"

cmd=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  python -u -m torch.distributed.run
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  tools/test_doscenes_full_scene_eval.py
  "${CONFIG}"
  "${CKPT}"
  --launcher pytorch
  --doscenes-csv "${DOSCENES_CSV}"
  --frame-stride "${FRAME_STRIDE}"
  --history-seconds "${HISTORY_SECONDS}"
  --planning-steps "${PLANNING_STEPS}"
  --drop-tail-frames "${DROP_TAIL_FRAMES}"
  --save-dir "${SAVE_DIR}"
  --index-json "${INDEX_JSON}"
  --metrics-json "${METRICS_JSON}"
  --cfg-options
  data.test.enable_doscenes_instruction=True
  "data.test.doscenes_csv=${DOSCENES_CSV}"
)

if [[ "${HISTORY_FRAMES}" != "0" ]]; then
  cmd+=(--history-frames "${HISTORY_FRAMES}")
fi

if [[ "${NO_INSTRUCTION}" == "1" ]]; then
  cmd+=(--no-instruction)
fi

if [[ "${RECORD_WARMUP_OUTPUTS}" == "1" ]]; then
  cmd+=(--record-warmup-outputs)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" FORCE_REWRITE_PREDS="${FORCE_REWRITE_PREDS}" "${cmd[@]}"

# DROP_TAIL_FRAMES=12 CUDA_VISIBLE_DEVICES=4,5,6,7 bash test/doscene_test_full_sc
# ene.sh
