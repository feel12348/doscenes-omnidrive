# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import pickle
import os
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import tqdm
from tools.data_converter import nuscenes_converter as nuscenes_converter
from mmdet3d.datasets import NuScenesDataset
import mmcv
from data_utils.trajectory_api import NuScenesTraj
from data_utils.nuscmap_extractor import NuscMapExtractor
from mmdet3d.core.bbox import Box3DMode
from openlanev2.centerline.io import io
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')
roi_size=(100, 50) # meters
cat2id_map={
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Add OmniDrive ego-coordinate GT, planning labels, maps and canbus to nuScenes info pkl.')
    parser.add_argument(
        '--data-root',
        default='./data/nuscenes/',
        help='nuScenes data root containing samples, maps and info pkl files.')
    parser.add_argument(
        '--can-bus-root',
        default=None,
        help='nuScenes canbus root. Defaults to --data-root.')
    parser.add_argument(
        '--skip-can-bus',
        action='store_true',
        help='Do not load nuScenes can_bus. Keep existing can_bus in input pkl, or fill zeros if missing.')
    parser.add_argument(
        '--lane-json-path',
        default='./data/nuscenes/data_dict_subset_B.json',
        help='OpenLane-V2 lane json path used to attach lane_info.')
    parser.add_argument(
        '--info-prefix',
        choices=['train', 'val', 'test'],
        default='val',
        help='Split suffix of the input info file.')
    parser.add_argument(
        '--input-info',
        default=None,
        help='Input pkl path. Defaults to <data-root>/nuscenes2d_temporal_infos_<info-prefix>.pkl.')
    parser.add_argument(
        '--output-info',
        default=None,
        help='Output pkl path. Defaults to <data-root>/nuscenes2d_ego_temporal_infos_<info-prefix><output-tag>.pkl.')
    parser.add_argument(
        '--output-tag',
        default='',
        help='Optional suffix appended before .pkl, e.g. _6s.')
    parser.add_argument(
        '--nuscenes-version',
        default='v1.0-trainval',
        help='nuScenes version used by the NuScenes API.')
    parser.add_argument(
        '--prediction-steps',
        type=int,
        default=12,
        help='Future agent prediction steps written to gt_fut_* fields.')
    parser.add_argument(
        '--planning-steps',
        type=int,
        default=6,
        help='Future ego planning steps written to gt_planning. 12 steps at 2Hz equals 6 seconds.')
    parser.add_argument(
        '--skip-agent-and-det-gt',
        action='store_true',
        help='Skip object trajectory and detection GT fields. This is enabled automatically for v1.0-test.')
    return parser.parse_args()

def _get_can_bus_info(nusc, nusc_can_bus, sample):
    if nusc_can_bus is None:
        return None
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(13)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')  # useless
    rotation = last_pose.pop('orientation')
    pos = last_pose.pop('pos')
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 13 elements
    return np.array(can_bus)

sample_dict = {}

def main():
    args = parse_args()
    data_root = args.data_root
    can_bus_root_path = args.can_bus_root or data_root
    input_info = args.input_info or os.path.join(
        data_root, 'nuscenes2d_temporal_infos_{}.pkl'.format(args.info_prefix))
    output_tag = args.output_tag
    output_info = args.output_info or os.path.join(
        data_root,
        'nuscenes2d_ego_temporal_infos_{}{}.pkl'.format(args.info_prefix, output_tag))

    key_infos = pickle.load(open(input_info, 'rb'))
    nuscenes = NuScenes(args.nuscenes_version, data_root)
    can_bus_dir = os.path.join(can_bus_root_path, 'can_bus')
    if args.skip_can_bus or not os.path.isdir(can_bus_dir):
        nusc_can_bus = None
        print(
            '[update_coords] CAN bus disabled or missing; keeping existing can_bus values when present.',
            flush=True)
    else:
        nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    skip_agent_and_det_gt = (
        args.skip_agent_and_det_gt
        or args.info_prefix == 'test'
        or 'test' in args.nuscenes_version
    )
    traj_api = NuScenesTraj(
        nuscenes,
        prediction_steps=args.prediction_steps,
        planning_steps=args.planning_steps,
        CLASSES=CLASSES,
        box_mode_3d=Box3DMode.LIDAR)
    map_api = NuscMapExtractor(data_root, roi_size)

    sample_dict = {}
    for split, segments in io.json_load(args.lane_json_path).items():
        for segment_id, timestamps in segments.items():
            for timestamp in timestamps:
                sample_dict[timestamp.split(sep='.')[0]] = (split, segment_id, timestamp.split(sep='.')[0])

    for current_id in tqdm.tqdm(range(len(key_infos['infos']))):
        sample = nuscenes.get('sample', key_infos['infos'][current_id]['token']) 
        info = key_infos['infos'][current_id]
        ego2global_rotation = info['ego2global_rotation']
        ego2global_translation = info['ego2global_translation']
        
        can_bus = _get_can_bus_info(nuscenes, nusc_can_bus, sample)
        if can_bus is not None:
            key_infos['infos'][current_id]["can_bus"] = can_bus
        elif 'can_bus' not in info:
            key_infos['infos'][current_id]["can_bus"] = np.zeros(13)
        # openlane
        if str(info['cams']['CAM_FRONT']['timestamp']) in sample_dict.keys():
            info['lane_info'] = sample_dict[str(info['cams']['CAM_FRONT']['timestamp'])]

        # motion prediction labels require sample annotations, which are not
        # available in nuScenes v1.0-test. The model test pipeline does not
        # consume these fields, so skip them for prediction-only test infos.
        if not skip_agent_and_det_gt:
            gt_fut_traj, gt_fut_yaw, gt_fut_traj_mask, gt_fut_idx = traj_api.get_traj_label(info['token'])
            info['gt_fut_traj'] = gt_fut_traj
            info['gt_fut_yaw'] = gt_fut_yaw
            info['gt_fut_traj_mask'] = gt_fut_traj_mask
            info['gt_fut_idx'] = gt_fut_idx

        # planning
        planning_all, planning_mask_all, command = traj_api.get_planning_label(info['token'])
        info['gt_planning'] = planning_all
        info['gt_planning_mask'] = planning_mask_all
        info['gt_planning_command'] = command
          
        # map
        scene_record = nuscenes.get('scene', sample['scene_token'])
        log_record = nuscenes.get('log', scene_record['log_token'])
        location = log_record['location']
        scene_name = scene_record['name']
        info['description'] = scene_record['description']
        info['location'] = location
        info['scene_name'] = scene_name
        map_geoms = map_api.get_map_geom(location, info['ego2global_translation'], info['ego2global_rotation'])

        map_label2geom = {}
        for k, v in map_geoms.items():
            if k in cat2id_map.keys():
                map_label2geom[cat2id_map[k]] = v
        info['map_geoms'] = map_label2geom

        if not skip_agent_and_det_gt:
            # detection, use ego coordinate
            ann_infos = list()
            for ann in sample['anns']:
                ann_info = nuscenes.get('sample_annotation', ann)
                velocity = nuscenes.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                if len(ann_info['attribute_tokens']) == 0:
                    ann_info['attr'] = ''
                else:
                    ann_info['attr'] = nuscenes.get('attribute', ann_info['attribute_tokens'][0])['name']
                ann_infos.append(ann_info)

            trans = -np.array(ego2global_translation)
            rot = Quaternion(ego2global_rotation).inverse
            gt_boxes = list()
            gt_velos = list()
            gt_names = list()
            gt_fullnames = list()
            num_lidar_pts = list()
            num_radar_pts = list()
            gt_valid_flags = list()
            gt_attrs= list()
            for ann_info in ann_infos:
                # Use ego coordinate.
                box = Box(
                    ann_info['translation'],
                    ann_info['size'],
                    Quaternion(ann_info['rotation']),
                    velocity=ann_info['velocity'],
                )
                box.translate(trans)
                box.rotate(rot)
                box_xyz = np.array(box.center)
                box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
                box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
                box_velo = np.array(box.velocity[:2])
                gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw])

                full_name = ann_info['category_name']
                if full_name in NuScenesDataset.NameMapping:
                    name = NuScenesDataset.NameMapping[full_name]

                valid_flag = ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] > 0

                gt_valid_flags.append(valid_flag)
                gt_names.append(name)
                gt_fullnames.append(full_name)
                gt_boxes.append(gt_box)
                gt_velos.append(box_velo)
                gt_attrs.append(ann_info['attr'])
                num_lidar_pts.append(ann_info['num_lidar_pts'])
                num_radar_pts.append(ann_info['num_radar_pts'])
            if len(gt_boxes) > 0:
                gt_valid_flags = np.array(gt_valid_flags, dtype=bool).reshape(-1)
                gt_boxes = np.concatenate(gt_boxes, axis=0).reshape(-1, 7)
                gt_velos = np.concatenate(gt_velos, axis=0).reshape(-1, 2)
                gt_names = np.array(gt_names)
                num_lidar_pts = np.array(num_lidar_pts)
                num_radar_pts = np.array(num_radar_pts)
            else:
                gt_valid_flags = np.array(gt_valid_flags, dtype=bool).reshape(-1)
                gt_boxes = np.array(gt_boxes).reshape(-1, 7)
                gt_velos = np.array(gt_velos).reshape(-1, 2)
                gt_names = np.array(gt_names)
                num_lidar_pts = np.array(num_lidar_pts)
                num_radar_pts = np.array(num_radar_pts)

            info['gt_boxes'] = gt_boxes
            info['gt_names'] = gt_names
            info['gt_velocity'] = gt_velos
            info['num_lidar_pts'] = num_lidar_pts
            info['num_radar_pts'] = num_radar_pts
            info['valid_flag'] = gt_valid_flags
            info['gt_fullnames'] = gt_fullnames
            info['gt_attrs'] = gt_attrs

        
        key_infos['infos'][current_id] = info

    mmcv.dump(key_infos, output_info)
    print(output_info)


if __name__ == '__main__':
    main()
