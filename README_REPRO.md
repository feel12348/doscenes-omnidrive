# doScenes Reproduction Package

This folder contains the code, configs, scripts, and small CSV annotation files
needed for the final doScenes 6s / 12-point `_new` training and official
150-scene test run.

Large runtime assets are intentionally not included:

- `.pth` checkpoints
- LLM / vision pretrained weights
- full nuScenes image data

The MMDetection3D source used by OmniDrive is included in this package under
`mmdetection3d/`.

Before running, provide those external assets under the relative paths expected
by the configs, for example:

```text
ckpts/eva02_petr_proj.pth
ckpts/pretrain_qformer/
data/nuscenes/
```

For testing, pass the trained checkpoint explicitly:

```bash
CHECKPOINT=/path/to/latest.pth bash scripts/doscenes_test_official150_new.sh
```

## Check Files

Check code-package files only:

```bash
bash check_repro_files.sh
```

After linking data and weights, check external assets too:

```bash
CHECK_EXTERNAL_ASSETS=1 bash check_repro_files.sh
```

## Train

```bash
bash scripts/doscenes_train_12pts_new.sh
```

Default training config:

```text
projects/configs/OmniDrive/mask_eva_lane_det_vlm_train_doscenes_only_6s_new.py
```

Default output:

```text
log_train/doscene_6s_12pts_new_fix_lm/
```

## Smoke Test

```bash
CHECKPOINT=/path/to/latest.pth GPUS=1 SMOKE_MAX_SCENES=1 SAVE_DIR=results_smoke_new \
bash scripts/doscenes_test_official150_new.sh
```

## Official 150-Scene Test

```bash
CHECKPOINT=/path/to/latest.pth GPUS=8 SAVE_DIR=results_doscenes_new_official150 \
bash scripts/doscenes_test_official150_new.sh
```

Submission CSV:

```text
results_doscenes_new_official150/submission.csv
```

The official test script runs one row per nuScenes test scene: first doScenes
instruction when present, and an empty instruction when a scene has no doScenes
instruction.
