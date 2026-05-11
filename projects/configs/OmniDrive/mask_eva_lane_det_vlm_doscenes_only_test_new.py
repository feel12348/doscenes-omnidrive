_base_ = ['./mask_eva_lane_det_vlm_doscenes_only_test.py']

# Test config matching checkpoints trained with StreamPETRHead_new /
# PETRTemporalTransformer_new. The challenge submit script controls the 12-point
# CSV output; this config controls model construction.
model = dict(
    enable_drivecode_numbers=False,
    pts_bbox_head=dict(
        type='StreamPETRHead_new',
        transformer=dict(
            type='PETRTemporalTransformer_new',
            flash_attn=True,
        ),
    ),
)
