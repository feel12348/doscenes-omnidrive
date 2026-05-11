_base_ = ['./mask_eva_lane_det_vlm_doscenes_only.py']

data_root = './data/nuscenes/'
doscenes_csv = 'data/annotated_doscenes_test.csv'

data = dict(
    val=dict(
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_test.pkl',
        doscenes_csv=doscenes_csv,
        only_doscenes_samples=True,
        enable_doscenes_instruction=True,
    ),
    test=dict(
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_test.pkl',
        doscenes_csv=doscenes_csv,
        only_doscenes_samples=True,
        enable_doscenes_instruction=True,
    ),
)

model = dict(
    save_path='./results_doscenes_test/',
)
