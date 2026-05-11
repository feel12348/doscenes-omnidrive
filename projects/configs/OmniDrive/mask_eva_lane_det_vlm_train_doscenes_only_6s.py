_base_ = ['./mask_eva_lane_det_vlm_train_doscenes_only.py']

data_root = './data/nuscenes/'
llm_path = 'ckpts/pretrain_qformer/'
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
collect_keys = [
    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp',
    'ego_pose', 'ego_pose_inv', 'command', 'can_bus'
]
drivecode_number_keys = []
ida_aug_conf = {
    "resize_lim": (0.37, 0.45),
    "final_dim": (320, 640),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}

train_pipeline_6s = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type='LoadAnnoatationVQA',
         base_vqa_path='./data/nuscenes/vqa/train/',
         base_desc_path='./data/nuscenes/desc/train/',
         base_conv_path='./data/nuscenes/conv/train/',
         base_key_path='./data/nuscenes/keywords/train/',
         tokenizer=llm_path,
         max_length=2048,
         ignore_type=[],
         lane_objs_info="./data/nuscenes/lane_obj_train.pkl",
         planning_steps=12),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D',
         keys=['lane_pts', 'input_ids', 'vlm_labels', 'gt_bboxes_3d',
               'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d',
               'depths', 'prev_exists'] + collect_keys + drivecode_number_keys,
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'scene_token', 'gt_bboxes_3d',
                    'gt_labels_3d'))
]

data = dict(
    train=dict(
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_train_6s.pkl',
        pipeline=train_pipeline_6s,
    ),
    val=dict(
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_val_6s.pkl',
    ),
    test=dict(
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_val_6s.pkl',
    ),
)
