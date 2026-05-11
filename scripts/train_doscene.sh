#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default to 4 GPUs: 0,1,2,3
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
GPUS="${GPUS:-4}"

# doScenes switches
ENABLE_DOSCENES="${ENABLE_DOSCENES:-1}"
RANDOM_DOSCENES="${RANDOM_DOSCENES:-1}"
ONLY_DOSCENES_SAMPLES="${ONLY_DOSCENES_SAMPLES:-0}"

bash "${REPO_ROOT}/omnidrive/tools/doscene_train.sh"

# CUDA_VISIBLE_DEVICES=2,3,4,5 GPUS=4 bash train/train_doscene.sh
