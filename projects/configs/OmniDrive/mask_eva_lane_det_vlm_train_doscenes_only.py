_base_ = ['./mask_eva_lane_det_vlm.py']

# Training-focused doScenes-only config.
# This config fixes train/val/test to only samples with valid doScenes
# instructions and always enables instruction injection.
doscenes_csv = 'data/annotated_doscenes.csv'
enable_doscenes_instruction = True
random_doscenes_instruction = True
only_doscenes_samples = True

data = dict(
    train=dict(
        doscenes_csv=doscenes_csv,
        enable_doscenes_instruction=enable_doscenes_instruction,
        random_doscenes_instruction=random_doscenes_instruction,
        only_doscenes_samples=only_doscenes_samples,
    ),
    val=dict(
        doscenes_csv=doscenes_csv,
        enable_doscenes_instruction=enable_doscenes_instruction,
        only_doscenes_samples=only_doscenes_samples,
    ),
    test=dict(
        doscenes_csv=doscenes_csv,
        enable_doscenes_instruction=enable_doscenes_instruction,
        only_doscenes_samples=only_doscenes_samples,
    ),
)
