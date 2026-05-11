# doScenes 12-Point Training And Test

This package contains code, configs, small CSV files, and the local
`mmdetection3d/` source used by OmniDrive. It intentionally does not include
`.pth` checkpoints, pretrained weights, or full nuScenes data.

Before running, provide external assets at the expected relative paths:

- `data/nuscenes/`
- `ckpts/eva02_petr_proj.pth`
- `ckpts/pretrain_qformer/`

For testing, pass the trained model path with `CHECKPOINT=/path/to/latest.pth`.

## Train

```bash
bash scripts/doscenes_train_12pts_new.sh
```

Equivalent explicit command:

```bash
CONFIG=projects/configs/OmniDrive/mask_eva_lane_det_vlm_train_doscenes_only_6s_new.py \
GPUS=8 \
WORK_DIR=log_train/doscene_6s_12pts_new_fix_lm \
DCSV=data/annotated_doscenes.csv \
bash scripts/doscenes_train_12pts_new.sh
```

Expected early training sanity check: `vlm_loss` / `vlm_ce` should be around
`1.x`, not stuck near `10.37`.

## Smoke Test

```bash
CHECKPOINT=/path/to/latest.pth \
SAVE_DIR=results_smoke_new \
GPUS=1 \
SMOKE_MAX_SCENES=1 \
bash scripts/doscenes_test_official150_new.sh
```

## Official 150-Scene Test

```bash
CHECKPOINT=/path/to/latest.pth \
SAVE_DIR=results_doscenes_new_official150 \
GPUS=8 \
bash scripts/doscenes_test_official150_new.sh
```

The official-format CSV is:

```text
results_doscenes_new_official150/submission.csv
```

The test script uses official mode: one row for each of 150 nuScenes test
scenes, first doScenes instruction when present, and empty instruction when no
instruction is available.
