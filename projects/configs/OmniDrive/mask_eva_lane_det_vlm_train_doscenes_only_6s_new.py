_base_ = ['./mask_eva_lane_det_vlm_train_doscenes_only_6s.py']

# 6s / 12-waypoint ordinary text training with the new StreamPETR temporal
# transformer. This does not enable DriveCode/number regression.
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
