import argparse
import csv
import importlib
import json
import os
import os.path as osp
import re
import sys
from collections import defaultdict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate doScenes challenge submission.csv from nuScenes test scenes.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--doscenes-csv',
        default='data/annotated_doscenes_test.csv',
        help='CSV with doScenes test instructions')
    parser.add_argument(
        '--save-dir',
        default='results_doscenes_challenge',
        help='directory for raw model outputs and metadata')
    parser.add_argument(
        '--submission-csv',
        default=None,
        help='official-format CSV path; default: <save-dir>/submission.csv')
    parser.add_argument(
        '--records-json',
        default=None,
        help='debug metadata JSON path; default: <save-dir>/prediction_records.json')
    parser.add_argument(
        '--instruction-mode',
        choices=['first', 'all'],
        default='first',
        help='first: one row per scene; all: run every instruction for each scene')
    parser.add_argument(
        '--include-scenes-without-instruction',
        action='store_true',
        help='include test scenes missing from doScenes CSV with an empty instruction')
    parser.add_argument(
        '--official-150',
        action='store_true',
        help='official mode: first instruction if available, empty otherwise, one row per test scene')
    parser.add_argument(
        '--anchor-offset',
        type=int,
        default=4,
        help='frame position used as first segment anchor; 4 means 2s history at 2Hz')
    parser.add_argument(
        '--history-frames',
        type=int,
        default=4,
        help='number of frames used to warm temporal memory before the anchor')
    parser.add_argument(
        '--planning-steps',
        type=int,
        default=12,
        help='number of future xy points required in submission')
    parser.add_argument(
        '--max-scenes',
        type=int,
        default=0,
        help='debug: process first N selected scenes only; 0 means all')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='build dataset and print selected rows without loading/running model')
    parser.add_argument(
        '--no-instruction',
        action='store_true',
        help='run matched history-only baseline by passing empty instructions')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=mmcv.DictAction,
        help='override config options, e.g. data.test.ann_file=...')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def setup_distributed(args):
    if args.launcher == 'none':
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 1, 0

    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    return dist.get_rank(), dist.get_world_size(), local_rank


def maybe_import_plugin(cfg, config_path):
    if not getattr(cfg, 'plugin', False):
        return
    if hasattr(cfg, 'plugin_dir'):
        module_dir = osp.dirname(cfg.plugin_dir)
    else:
        module_dir = osp.dirname(config_path)
    module_path = module_dir.replace('/', '.').strip('.')
    if module_path:
        importlib.import_module(module_path)


def prepare_cfg(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    maybe_import_plugin(cfg, args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    if not isinstance(cfg.data.test, dict):
        raise TypeError('This script expects cfg.data.test to be a dict.')
    cfg.data.test.test_mode = True
    # The submission script owns scene/instruction filtering. Some doScenes
    # configs keep only scenes present in the CSV, which is useful locally but
    # can silently drop official test scenes.
    cfg.data.test.only_doscenes_samples = False
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    return cfg


def load_instruction_map(csv_path):
    scene_to_insts = defaultdict(list)
    if not csv_path:
        return scene_to_insts
    if not osp.isabs(csv_path):
        csv_path = osp.join(os.getcwd(), csv_path)
    if not osp.exists(csv_path):
        raise FileNotFoundError(csv_path)

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_id, row in enumerate(reader):
            scene_number = row.get('Scene Number')
            instruction = (row.get('Instruction') or '').strip()
            if not scene_number or not instruction:
                continue
            try:
                scene_name = f"scene-{int(float(scene_number)):04d}"
            except ValueError:
                continue
            scene_to_insts[scene_name].append(
                dict(
                    instruction_id=len(scene_to_insts[scene_name]),
                    instruction=instruction,
                    instruction_type=(row.get('Instruction Type') or 'unknown').strip() or 'unknown',
                    annotator=(row.get('Annotator') or '').strip(),
                    source_row=row.get('Source Row') or row_id,
                )
            )
    return scene_to_insts


def build_scene_indices(dataset):
    scene_to_indices = defaultdict(list)
    for idx, info in enumerate(dataset.data_infos):
        scene_to_indices[info.get('scene_name', '')].append(idx)
    for scene_name, indices in scene_to_indices.items():
        indices.sort(key=lambda i: dataset.data_infos[i].get('frame_idx', i))
    return scene_to_indices


def select_tasks(dataset, scene_to_insts, args):
    scene_to_indices = build_scene_indices(dataset)
    selected_scenes = sorted(scene_to_indices)
    if not args.include_scenes_without_instruction:
        selected_scenes = [s for s in selected_scenes if s in scene_to_insts]
    if args.max_scenes > 0:
        selected_scenes = selected_scenes[:args.max_scenes]

    tasks = []
    skipped_short = []
    for scene_name in selected_scenes:
        indices = scene_to_indices[scene_name]
        if not indices:
            continue
        anchor_pos = min(max(args.anchor_offset, 0), len(indices) - 1)
        if anchor_pos < args.history_frames:
            skipped_short.append(scene_name)
            continue
        anchor_index = indices[anchor_pos]
        history_indices = indices[anchor_pos - args.history_frames: anchor_pos]
        instructions = scene_to_insts.get(scene_name, [])
        if not instructions:
            instructions = [dict(
                instruction_id=0,
                instruction='',
                instruction_type='missing',
                annotator='',
                source_row='',
            )]
        elif args.instruction_mode == 'first':
            instructions = instructions[:1]

        for inst in instructions:
            tasks.append(dict(
                scene_name=scene_name,
                anchor_index=anchor_index,
                history_indices=history_indices,
                instruction_id=inst['instruction_id'],
                instruction=inst['instruction'],
                instruction_type=inst['instruction_type'],
                annotator=inst.get('annotator', ''),
                source_row=inst.get('source_row', ''),
            ))
    return tasks, skipped_short, scene_to_indices


def build_model_for_test(cfg, checkpoint_path, dataset, save_dir, device_id=0):
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    model.save_path = save_dir
    model = MMDataParallel(model.cuda(device_id), device_ids=[device_id])
    model.eval()
    return model


def gather_records(records, rank, world_size):
    if world_size == 1:
        return records
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, records)
    if rank != 0:
        return None
    merged = []
    for part in gathered:
        merged.extend(part)
    merged.sort(key=lambda r: (r['scene_name'], r['instruction_id'], r['task_id']))
    return merged


def reset_temporal_memory(model):
    module = model.module if hasattr(model, 'module') else model
    if hasattr(module, 'pts_bbox_head') and getattr(module, 'with_pts_bbox', False):
        module.pts_bbox_head.reset_memory()
    if hasattr(module, 'map_head') and getattr(module, 'with_map_head', False):
        module.map_head.reset_memory()
    module.test_flag = True


def inject_runtime_meta(batch, skip_save=False, output_name=None):
    if 'img_metas' not in batch:
        return

    def _inject(obj):
        if hasattr(obj, 'data'):
            _inject(obj.data)
        elif isinstance(obj, dict):
            if 'sample_idx' in obj:
                obj['skip_save'] = bool(skip_save)
                if output_name is not None:
                    obj['output_name'] = output_name
            for value in obj.values():
                _inject(value)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _inject(item)

    _inject(batch['img_metas'])


def run_single_sample(model, dataset, index, instruction, output_name=None, skip_save=False):
    scene_name = dataset.data_infos[index].get('scene_name', '')
    if hasattr(dataset, 'doscenes_map'):
        dataset.doscenes_map[scene_name] = instruction
    sample = dataset[index]
    batch = collate([sample], samples_per_gpu=1)
    inject_runtime_meta(batch, skip_save=skip_save, output_name=output_name)
    with torch.no_grad():
        return model(return_loss=False, rescale=True, **batch)


def parse_pt_traj_from_text(text):
    match = re.search(r'\[PT, \((\+?[\d\.-]+, \+?[\d\.-]+)\)(, \(\+?[\d\.-]+, \+?[\d\.-]+\))*\]', text)
    if not match:
        return None
    coords = re.findall(r'\(\+?[\d\.-]+, \+?[\d\.-]+\)', match.group(0))
    points = []
    for coord in coords:
        nums = re.findall(r'-?\d+\.\d+|-?\d+', coord)
        if len(nums) < 2:
            return None
        points.append((float(nums[0]), float(nums[1])))
    return np.array(points, dtype=np.float32) if points else None


def extract_answer_text(text_out):
    if not isinstance(text_out, list):
        return None
    for item in text_out:
        if not isinstance(item, dict):
            continue
        answer = item.get('A')
        if isinstance(answer, list) and answer:
            return answer[0]
        if isinstance(answer, str):
            return answer
    return None


def parse_prediction_from_model_output(outputs):
    if not isinstance(outputs, list) or not outputs or not isinstance(outputs[0], dict):
        return None, None, 'prediction_invalid_output'
    text_out = outputs[0].get('text_out')
    answer_text = extract_answer_text(text_out)
    if not isinstance(answer_text, str):
        return None, text_out, 'prediction_missing_answer'
    traj = parse_pt_traj_from_text(answer_text)
    if traj is None:
        return None, text_out, 'prediction_parse_failed'
    return traj, text_out, None


def pad_or_trim_traj(traj, steps):
    if traj is None or len(traj) == 0:
        return np.zeros((steps, 2), dtype=np.float32), False
    traj = np.asarray(traj, dtype=np.float32)[:, :2]
    if traj.shape[0] >= steps:
        return traj[:steps], True
    padded = np.zeros((steps, 2), dtype=np.float32)
    padded[:traj.shape[0]] = traj
    padded[traj.shape[0]:] = traj[-1]
    return padded, False


def safe_name(text, max_len=48):
    text = re.sub(r'[^0-9a-zA-Z_-]+', '_', str(text or '').strip())
    text = re.sub(r'_+', '_', text).strip('_')
    return (text or 'empty')[:max_len]


def write_submission_csv(path, records, planning_steps):
    mmcv.mkdir_or_exist(osp.dirname(path) or '.')
    header = ['sample_token', 'instruction']
    for step in range(1, planning_steps + 1):
        header.extend([f'x{step}', f'y{step}'])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for record in records:
            row = {
                'sample_token': record['sample_token'],
                'instruction': record['instruction'],
            }
            traj = record['pred_traj_xy']
            for step, (x, y) in enumerate(traj, start=1):
                row[f'x{step}'] = f'{float(x):.6f}'
                row[f'y{step}'] = f'{float(y):.6f}'
            writer.writerow(row)


def main():
    args = parse_args()
    if args.official_150:
        args.instruction_mode = 'first'
        args.include_scenes_without_instruction = True

    rank, world_size, local_rank = setup_distributed(args)
    cfg = prepare_cfg(args)
    dataset = build_dataset(cfg.data.test)
    scene_to_insts = load_instruction_map(args.doscenes_csv)
    tasks, skipped_short, scene_to_indices = select_tasks(dataset, scene_to_insts, args)

    submission_csv = args.submission_csv or osp.join(args.save_dir, 'submission.csv')
    records_json = args.records_json or osp.join(args.save_dir, 'prediction_records.json')

    if rank == 0:
        print(
            '[doScenesSubmit] '
            f'dataset_samples={len(dataset)}, scenes={len(scene_to_indices)}, '
            f'instruction_scenes={len(scene_to_insts)}, tasks={len(tasks)}, '
            f'instruction_mode={args.instruction_mode}, anchor_offset={args.anchor_offset}, '
            f'history_frames={args.history_frames}, world_size={world_size}'
        )
        if skipped_short:
            print(f'[doScenesSubmit] skipped_short_scenes={len(skipped_short)}')
        if args.instruction_mode == 'all':
            print('[doScenesSubmit] warning: all mode may create duplicate sample_token rows; use it for instruction-level testing, not strict official submission.')
    if args.dry_run:
        if rank == 0:
            for task in tasks[:10]:
                info = dataset.data_infos[task['anchor_index']]
                print(
                    '[dry-run] '
                    f"{task['scene_name']} scene_token={info.get('scene_token')} "
                    f"frame_idx={info.get('frame_idx')} inst={task['instruction_id']} "
                    f"instruction={task['instruction'][:100]!r}"
                )
            print(f'[dry-run] would write: {submission_csv}')
        return

    if rank == 0:
        mmcv.mkdir_or_exist(args.save_dir)
    if world_size > 1:
        dist.barrier()
    model = build_model_for_test(cfg, args.checkpoint, dataset, args.save_dir, device_id=local_rank)

    records = []
    for task_id, task in enumerate(tasks):
        if task_id % world_size != rank:
            continue
        info = dataset.data_infos[task['anchor_index']]
        scene_name = task['scene_name']
        scene_token = info.get('scene_token', '')
        instruction = '' if args.no_instruction else task['instruction']
        output_name = (
            f"{safe_name(scene_name)}__inst{int(task['instruction_id']):03d}"
            f"__frame{int(info.get('frame_idx', task['anchor_index'])):04d}.json"
        )

        reset_temporal_memory(model)
        for warm_idx in task['history_indices']:
            run_single_sample(model, dataset, warm_idx, instruction='', skip_save=True)

        outputs = run_single_sample(
            model,
            dataset,
            task['anchor_index'],
            instruction=instruction,
            output_name=output_name,
            skip_save=False,
        )
        pred_traj, text_out, pred_err = parse_prediction_from_model_output(outputs)
        pred_traj, full_length = pad_or_trim_traj(pred_traj, args.planning_steps)

        records.append(dict(
            task_id=task_id,
            sample_token=scene_token,
            scene_name=scene_name,
            anchor_sample_token=info.get('token', ''),
            frame_idx=int(info.get('frame_idx', task['anchor_index'])),
            instruction_id=int(task['instruction_id']),
            instruction=instruction,
            instruction_type=task['instruction_type'],
            output_name=output_name,
            prediction_error=pred_err,
            prediction_full_length=bool(full_length),
            pred_traj_xy=pred_traj.tolist(),
            text_out=text_out,
        ))

        if len(records) % 10 == 0:
            print(f'[doScenesSubmit][rank{rank}] local_completed={len(records)} global_task={task_id + 1}/{len(tasks)}', flush=True)

    all_records = gather_records(records, rank, world_size)
    if rank == 0:
        write_submission_csv(submission_csv, all_records, args.planning_steps)
        mmcv.dump(all_records, records_json)
        print(f'[doScenesSubmit] wrote submission: {submission_csv}')
        print(f'[doScenesSubmit] wrote records: {records_json}')
        print(f'[doScenesSubmit] rows={len(all_records)}')

    if world_size > 1:
        dist.barrier()


if __name__ == '__main__':
    main()
