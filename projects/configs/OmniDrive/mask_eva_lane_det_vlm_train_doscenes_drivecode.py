_base_ = ['./mask_eva_lane_det_vlm_train_doscenes_only.py']

# DriveCode-style trajectory-number training.
# This keeps the original doScenes-only training setup, but replaces only the
# planning trajectory waypoint numbers with <number_token> and supervises them
# through the continuous number regression head.
enable_drivecode_numbers = True
drivecode_number_token = '<number_token>'
drivecode_number_token_id = None
drivecode_number_keys = ['number_values']

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
llm_path = 'ckpts/pretrain_qformer/'
collect_keys = [
    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp',
    'ego_pose', 'ego_pose_inv', 'command', 'can_bus'
]
ida_aug_conf = {
    "resize_lim": (0.37, 0.45),
    "final_dim": (320, 640),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}

model = dict(
    enable_drivecode_numbers=enable_drivecode_numbers,
    drivecode_number_token=drivecode_number_token,
    drivecode_number_token_id=drivecode_number_token_id,
)

train_pipeline = [
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
         enable_drivecode_numbers=enable_drivecode_numbers,
         drivecode_number_token=drivecode_number_token,
         drivecode_number_token_id=drivecode_number_token_id),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D',
         keys=['lane_pts', 'input_ids', 'vlm_labels', 'number_values',
               'gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels',
               'centers2d', 'depths', 'prev_exists'] + collect_keys,
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'scene_token', 'gt_bboxes_3d',
                    'gt_labels_3d'))
]

data = dict(
    train=dict(pipeline=train_pipeline),
)
