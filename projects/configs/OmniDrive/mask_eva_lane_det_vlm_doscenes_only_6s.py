_base_ = ['./mask_eva_lane_det_vlm_doscenes_only.py']

data_root = './data/nuscenes/'

data = dict(
    val=dict(
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_val_6s.pkl',
    ),
    test=dict(
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_val_6s.pkl',
    ),
)
