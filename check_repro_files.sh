#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

required_files=(
  "README_REPRO.md"
  "DOSCENES_RUN.md"
  "requirements.txt"
  "update_coords.py"
  "mmdetection3d/setup.py"
  "mmdetection3d/mmdet3d/__init__.py"
  "mmdetection3d/configs/_base_/datasets/nus-3d.py"
  "mmdetection3d/configs/_base_/default_runtime.py"
  "tools/doscene_train.sh"
  "tools/dist_train.sh"
  "tools/test_doscenes_challenge_submit.py"
  "scripts/doscenes_train_12pts_new.sh"
  "scripts/doscenes_test_official150_new.sh"
  "projects/configs/OmniDrive/mask_eva_lane_det_vlm_train_doscenes_only_6s.py"
  "projects/configs/OmniDrive/mask_eva_lane_det_vlm_train_doscenes_only_6s_new.py"
  "projects/configs/OmniDrive/mask_eva_lane_det_vlm_doscenes_only_test.py"
  "projects/configs/OmniDrive/mask_eva_lane_det_vlm_doscenes_only_test_new.py"
  "projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py"
  "projects/mmdet3d_plugin/models/detectors/petr3d.py"
  "projects/mmdet3d_plugin/models/dense_heads/llava_llama.py"
  "projects/mmdet3d_plugin/models/utils/misc.py"
  "data/annotated_doscenes.csv"
  "data/annotated_doscenes_all.csv"
  "data/annotated_doscenes_test.csv"
)

external_files=(
  "data/nuscenes/nuscenes2d_ego_temporal_infos_train_6s.pkl"
  "data/nuscenes/nuscenes2d_ego_temporal_infos_val_6s.pkl"
  "data/nuscenes/nuscenes2d_ego_temporal_infos_test.pkl"
  "data/nuscenes/can_bus"
  "ckpts/eva02_petr_proj.pth"
  "ckpts/pretrain_qformer/config.json"
)

if [[ "${CHECK_EXTERNAL_ASSETS:-0}" == "1" ]]; then
  required_files+=("${external_files[@]}")
fi

missing=0
for path in "${required_files[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[missing] ${path}"
    missing=1
  else
    echo "[ok] ${path}"
  fi
done

if [[ "${missing}" != "0" ]]; then
  echo "[check] failed: some required files are missing" >&2
  exit 1
fi

if [[ "${CHECK_EXTERNAL_ASSETS:-0}" == "1" ]]; then
  echo "[check] all code files and external assets are present"
else
  echo "[check] all code files are present"
  echo "[check] external data/weights were skipped; set CHECK_EXTERNAL_ASSETS=1 to check them"
fi
