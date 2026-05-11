#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-projects/configs/OmniDrive/mask_eva_lane_det_vlm_train_doscenes_only_6s_new.py}"
GPUS="${GPUS:-8}"
WORK_DIR="${WORK_DIR:-log_train/doscene_6s_12pts_new_fix_lm}"
DCSV="${DCSV:-data/annotated_doscenes.csv}"

required_files=(
  "${CONFIG}"
  "tools/doscene_train.sh"
  "tools/dist_train.sh"
  "${DCSV}"
  "data/nuscenes/nuscenes2d_ego_temporal_infos_train_6s.pkl"
  "data/nuscenes/nuscenes2d_ego_temporal_infos_val_6s.pkl"
  "ckpts/eva02_petr_proj.pth"
  "ckpts/pretrain_qformer/config.json"
)

for path in "${required_files[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[doScenesTrain][ERROR] missing required file: ${path}" >&2
    exit 1
  fi
done

echo "[doScenesTrain] CONFIG=${CONFIG}"
echo "[doScenesTrain] GPUS=${GPUS}"
echo "[doScenesTrain] WORK_DIR=${WORK_DIR}"
echo "[doScenesTrain] DCSV=${DCSV}"

CONFIG="${CONFIG}" \
GPUS="${GPUS}" \
WORK_DIR="${WORK_DIR}" \
DCSV="${DCSV}" \
ENABLE_DOSCENES=1 \
RANDOM_DOSCENES=1 \
ONLY_DOSCENES_SAMPLES=1 \
bash tools/doscene_train.sh
