import argparse
import csv
import importlib
import json
import glob
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

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description='doScenes sliding-window memory-aware evaluation')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--doscenes-csv',
        default='data/annotated_doscenes.csv',
        help='CSV with doScenes instructions')
    parser.add_argument(
        '--save-dir',
        default=None,
        help='override model.save_path (directory for generated text)')
    parser.add_argument(
        '--pred-record-json',
        default=None,
        help='optional path to dump per-window prediction records')
    parser.add_argument(
        '--single-window-output-json',
        default=None,
        help='write all window-level raw outputs into one json file (recommended for large runs)')
    parser.add_argument(
        '--per-instruction-json-dir',
        default=None,
        help='if set, save one json per instruction (all its windows) and compute metrics by reading these json files')
    parser.add_argument(
        '--compute-ade-per-window',
        action='store_true',
        help='if set, compute ADE immediately for each window; default is compute once at final aggregation')
    parser.add_argument(
        '--metrics-json',
        default=None,
        help='optional path to dump final metrics json')
    parser.add_argument('--fps', type=float, default=2.0, help='frame rate for window conversion')
    parser.add_argument('--history-seconds', type=float, default=2.0, help='history seconds (memory warm-up)')
    parser.add_argument('--future-seconds', type=float, default=6.0, help='future seconds (window validity)')
    parser.add_argument(
        '--obs-len',
        type=int,
        default=0,
        help='openemma-style history length in frames; if >0 overrides history-seconds')
    parser.add_argument(
        '--fut-len',
        type=int,
        default=0,
        help='openemma-style future length in frames; if >0 overrides future-seconds')
    parser.add_argument(
        '--planning-steps',
        type=int,
        default=6,
        help='number of future trajectory waypoints required for metrics; use 12 for 6s at 2Hz')
    parser.add_argument('--window-stride', type=int, default=1, help='sliding window stride in frames')
    parser.add_argument('--max-scenes', type=int, default=0, help='debug: process first N scenes only (0 means all)')
    parser.add_argument(
        '--max-windows-per-scene',
        type=int,
        default=0,
        help='debug: process first N windows per scene only (0 means all)')
    parser.add_argument(
        '--record-history-outputs',
        action='store_true',
        help='if set, also save model outputs of history warm-up frames')
    parser.add_argument(
        '--no-instruction',
        action='store_true',
        help='if set, ignore doScenes language instruction and use empty text')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=mmcv.DictAction,
        help='override some settings in config')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher for multi-gpu evaluation')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def load_instruction_map(csv_path):
    scene_to_insts = defaultdict(list)
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_number = row.get('Scene Number')
            instruction_type = (row.get('Instruction Type') or 'unknown').strip() or 'unknown'
            instruction = (row.get('Instruction') or '').strip()
            if not scene_number or not instruction:
                continue
            try:
                scene_name = f"scene-{int(float(scene_number)):04d}"
            except ValueError:
                continue
            scene_to_insts[scene_name].append(
                dict(
                    instruction=instruction,
                    instruction_type=instruction_type,
                )
            )
    return scene_to_insts


def maybe_import_plugin(cfg, config_path):
    if not hasattr(cfg, 'plugin') or not cfg.plugin:
        return
    if hasattr(cfg, 'plugin_dir'):
        module_dir = os.path.dirname(cfg.plugin_dir)
    else:
        module_dir = os.path.dirname(config_path)
    module_parts = module_dir.split('/')
    module_path = module_parts[0]
    for part in module_parts[1:]:
        if part:
            module_path = module_path + '.' + part
    importlib.import_module(module_path)


def prepare_cfg(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    maybe_import_plugin(cfg, args.config)
    cfg.model.pretrained = None
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    else:
        raise TypeError('Only dict cfg.data.test is supported in this script.')
    return cfg


def build_model_and_dataset(cfg, checkpoint_path, save_dir=None, device_id=0):
    dataset = build_dataset(cfg.data.test)
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    if save_dir is not None:
        model.save_path = save_dir
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(device_id), device_ids=[device_id])
    model.eval()
    return model, dataset


def setup_distributed(args):
    if args.launcher == 'none':
        torch.cuda.set_device(0)
        return 0, 1, 0

    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank


def is_main_process(rank):
    return rank == 0


def gather_records(records, rank, world_size):
    if world_size == 1:
        return records
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, records)
    if rank == 0:
        merged = []
        for part in gathered:
            merged.extend(part)
        return merged
    return None


def gather_error_stats(error_stats, rank, world_size):
    if world_size == 1:
        return dict(error_stats)
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, dict(error_stats))
    if rank == 0:
        merged = defaultdict(int)
        for part in gathered:
            for k, v in part.items():
                merged[k] += int(v)
        return dict(merged)
    return None


def sanitize_instruction_text(text, max_len=48):
    s = re.sub(r'[^0-9a-zA-Z_-]+', '_', (text or '').strip())
    s = re.sub(r'_+', '_', s).strip('_')
    if not s:
        s = 'empty'
    return s[:max_len]


def load_records_from_instruction_json_dir(json_dir):
    all_records = []
    paths = sorted(glob.glob(osp.join(json_dir, '*.json')))
    for p in paths:
        try:
            data = mmcv.load(p)
        except Exception:
            continue
        if isinstance(data, dict):
            recs = data.get('records', [])
            if isinstance(recs, list):
                all_records.extend(recs)
        elif isinstance(data, list):
            all_records.extend(data)
    return all_records, paths


def reset_temporal_memory(model):
    module = model.module if hasattr(model, 'module') else model
    if hasattr(module, 'pts_bbox_head') and module.with_pts_bbox:
        module.pts_bbox_head.reset_memory()
    if hasattr(module, 'map_head') and module.with_map_head:
        module.map_head.reset_memory()
    module.test_flag = True


def inject_runtime_meta(batch, skip_save=False, output_name=None):
    if 'img_metas' not in batch:
        return

    def _inject(obj):
        if hasattr(obj, 'data'):
            _inject(obj.data)
            return
        if isinstance(obj, dict):
            if 'sample_idx' in obj:
                obj['skip_save'] = bool(skip_save)
                if output_name is not None:
                    obj['output_name'] = output_name
            for value in obj.values():
                _inject(value)
            return
        if isinstance(obj, list):
            for item in obj:
                _inject(item)
            return
        if isinstance(obj, tuple):
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
        outputs = model(return_loss=False, rescale=True, **batch)
    return outputs


def build_scene_indices(dataset):
    scene_to_indices = defaultdict(list)
    for idx, info in enumerate(dataset.data_infos):
        scene_to_indices[info.get('scene_name', '')].append(idx)
    for scene_name, indices in scene_to_indices.items():
        indices.sort(key=lambda i: dataset.data_infos[i].get('frame_idx', i))
        scene_to_indices[scene_name] = indices
    return scene_to_indices


def parse_pt_traj_from_text(text):
    full_match = re.search(r'\[PT, \((\+?[\d\.-]+, \+?[\d\.-]+)\)(, \(\+?[\d\.-]+, \+?[\d\.-]+\))*\]', text)
    if not full_match:
        return None
    coordinates_matches = re.findall(r'\(\+?[\d\.-]+, \+?[\d\.-]+\)', full_match.group(0))
    points = []
    for coord in coordinates_matches:
        nums = re.findall(r'-?\d+\.\d+|-?\d+', coord)
        if len(nums) < 2:
            return None
        points.append((float(nums[0]), float(nums[1])))
    if not points:
        return None
    return np.array(points, dtype=np.float32)


def parse_prediction_from_file(path):
    if not osp.exists(path):
        return None, 'prediction_file_missing', None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
    except Exception:
        return None, 'prediction_file_invalid_json', None

    if not isinstance(pred_data, list) or not pred_data:
        return None, 'prediction_empty', pred_data

    answer_text = None
    for item in pred_data:
        if isinstance(item, dict):
            a = item.get('A')
            if isinstance(a, list) and a:
                answer_text = a[0]
                break
            if isinstance(a, str):
                answer_text = a
                break

    if not isinstance(answer_text, str):
        return None, 'prediction_missing_answer', pred_data

    traj = parse_pt_traj_from_text(answer_text)
    if traj is None:
        return None, 'prediction_parse_failed', pred_data
    return traj, None, pred_data


def parse_prediction_from_model_output(outputs):
    if outputs is None:
        return None, 'prediction_empty', None
    if not isinstance(outputs, list) or len(outputs) == 0:
        return None, 'prediction_invalid_output_type', None
    sample_out = outputs[0]
    if not isinstance(sample_out, dict):
        return None, 'prediction_invalid_output_payload', None
    text_out = sample_out.get('text_out', None)
    if not isinstance(text_out, list) or len(text_out) == 0:
        return None, 'prediction_missing_answer', text_out

    answer_text = None
    for item in text_out:
        if isinstance(item, dict):
            a = item.get('A')
            if isinstance(a, list) and len(a) > 0:
                answer_text = a[0]
                break
            if isinstance(a, str):
                answer_text = a
                break
    if not isinstance(answer_text, str):
        return None, 'prediction_missing_answer', text_out
    traj = parse_pt_traj_from_text(answer_text)
    if traj is None:
        return None, 'prediction_parse_failed', text_out
    return traj, None, text_out


def metric_keys(planning_steps):
    return [f'ade{step // 2}s' for step in range(2, planning_steps + 1, 2)]


def format_metrics(metrics):
    keys = metric_keys(max(2, 2 * len([k for k in metrics if k.startswith('ade') and k.endswith('s')])))
    keys = [k for k in keys if k in metrics]
    parts = [f"{k.upper()}={metrics[k]}" for k in keys]
    parts.append(f"MADE={metrics.get('made')}")
    return ', '.join(parts)


def compute_ades(pred_xy, gt_xy, planning_steps=6):
    if pred_xy.shape[0] < planning_steps or gt_xy.shape[0] < planning_steps:
        return None

    d = np.sqrt(np.sum((pred_xy[:planning_steps, :2] - gt_xy[:planning_steps, :2]) ** 2, axis=-1))
    metrics = {}
    for step in range(2, planning_steps + 1, 2):
        metrics[f'ade{step // 2}s'] = float(np.mean(d[:step]))
    metrics['made'] = float(np.mean([metrics[k] for k in metric_keys(planning_steps)]))
    return metrics


def summarize_metric_records(records, planning_steps=6):
    valid = [r for r in records if r.get('valid_for_metric', False)]
    keys = metric_keys(planning_steps)
    if not valid:
        return dict(count=0, **{k: None for k in keys}, made=None)

    summary = {k: float(np.mean([r[k] for r in valid])) for k in keys}
    summary['made'] = float(np.mean([summary[k] for k in keys]))
    summary['count'] = len(valid)
    return summary


def summarize_error_records(records):
    total_records = len(records)
    error_records = [r for r in records if not r.get('valid_for_metric', False)]
    error_count = len(error_records)
    error_stats = defaultdict(int)
    for record in error_records:
        error_stats[record.get('prediction_error', 'unknown')] += 1
    error_rate = (float(error_count) / float(total_records)) if total_records > 0 else 0.0
    return dict(
        total_records=total_records,
        error_count=error_count,
        error_rate=error_rate,
        error_stats=dict(error_stats),
    )


def finalize_ade_for_records(records, planning_steps=6):
    extra_errors = defaultdict(int)
    keys = metric_keys(planning_steps)
    for record in records:
        if not record.get('valid_for_metric', False):
            continue
        if all(k in record for k in keys + ['made']):
            continue
        pred = np.array(record.get('pred_traj_xy', []), dtype=np.float32)
        gt = np.array(record.get('gt_traj_xy', []), dtype=np.float32)
        metrics = compute_ades(pred, gt, planning_steps)
        if metrics is None:
            record['valid_for_metric'] = False
            record['prediction_error'] = 'prediction_too_short'
            extra_errors['prediction_too_short'] += 1
            continue
        record.update(metrics)
    return extra_errors


def log_window_result(rank, record):
    prefix = (
        f"[SlidingEval][rank{rank}] "
        f"{record['scene_name']} inst={record['instruction_id']} "
        f"win={record['window_id']} frame={record['frame_idx']} "
        f"token={record['sample_token']}"
    )
    if record.get('valid_for_metric', False):
        ade_keys = [k for k in sorted(record.keys()) if k.startswith('ade') and k.endswith('s')]
        if ade_keys and 'made' in record:
            values = ' '.join(f"{k.upper()}={record[k]:.4f}" for k in ade_keys)
            print(
                prefix + f" | {values} MADE={record['made']:.4f}",
                flush=True,
            )
        else:
            print(prefix + " | done", flush=True)
    else:
        print(prefix + f" | error={record.get('prediction_error', 'unknown')}", flush=True)


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed(args)

    if args.window_stride < 1:
        raise ValueError('--window-stride must be >= 1')
    if args.fps <= 0:
        raise ValueError('--fps must be > 0')

    history_frames = int(round(args.history_seconds * args.fps))
    future_frames = int(round(args.future_seconds * args.fps))
    if args.obs_len > 0:
        history_frames = args.obs_len
    if args.fut_len > 0:
        future_frames = args.fut_len
    if history_frames < 1:
        raise ValueError('history_frames must be >= 1; adjust --history-seconds or --fps')
    if future_frames < 1:
        raise ValueError('future_frames must be >= 1; adjust --future-seconds or --fps')

    cfg = prepare_cfg(args)
    model, dataset = build_model_and_dataset(
        cfg, args.checkpoint, save_dir=args.save_dir, device_id=local_rank
    )

    if args.save_dir is None:
        save_dir = model.module.save_path
    else:
        save_dir = args.save_dir
    mmcv.mkdir_or_exist(save_dir)
    if args.per_instruction_json_dir:
        mmcv.mkdir_or_exist(args.per_instruction_json_dir)

    scene_to_insts = load_instruction_map(args.doscenes_csv)
    scene_to_indices = build_scene_indices(dataset)

    selected_scenes = [s for s in scene_to_indices.keys() if s in scene_to_insts]
    selected_scenes.sort()
    if args.max_scenes > 0:
        selected_scenes = selected_scenes[:args.max_scenes]

    if is_main_process(rank):
        print(
            f'[SlidingEval] world_size={world_size}, scenes={len(selected_scenes)}, fps={args.fps}, '
            f'history={args.history_seconds}s({history_frames}f), '
            f'future={args.future_seconds}s({future_frames}f), stride={args.window_stride}'
        )

    records = []
    local_errors = defaultdict(int)
    instruction_task_id = 0

    for sidx, scene_name in enumerate(selected_scenes):
        frame_indices = scene_to_indices[scene_name]
        instructions = scene_to_insts[scene_name]

        # Window center is current frame to be evaluated.
        # Must have enough history and enough future inside the scene window.
        min_pos = history_frames
        max_pos = len(frame_indices) - future_frames - 1
        if max_pos < min_pos:
            if is_main_process(rank):
                print(
                    f'[SlidingEval] scene {scene_name} skipped: '
                    f'frames={len(frame_indices)} not enough for window ({history_frames}+1+{future_frames})'
                )
            continue

        target_positions = list(range(min_pos, max_pos + 1, args.window_stride))
        if args.max_windows_per_scene > 0:
            target_positions = target_positions[:args.max_windows_per_scene]

        if is_main_process(rank):
            print(
                f'[SlidingEval] scene {sidx + 1}/{len(selected_scenes)} {scene_name}: '
                f'windows={len(target_positions)}, instructions={len(instructions)}'
            )

        for inst_id, instruction in enumerate(instructions):
            instruction_text = instruction.get('instruction', '') if isinstance(instruction, dict) else str(instruction)
            instruction_type = instruction.get('instruction_type', 'unknown') if isinstance(instruction, dict) else 'unknown'
            process_this_instruction = (instruction_task_id % world_size == rank)
            instruction_task_id += 1
            if not process_this_instruction:
                continue

            inst_records = []
            inst_error_counts = defaultdict(int)
            for win_id, target_pos in enumerate(target_positions):
                target_index = frame_indices[target_pos]
                history_indices = frame_indices[target_pos - history_frames: target_pos]

                # Reset and warm up 2s history (4 frames @2Hz by default).
                reset_temporal_memory(model)
                for hidx, warm_idx in enumerate(history_indices):
                    prefix_output_name = None
                    prefix_skip_save = True
                    if args.record_history_outputs:
                        warm_info = dataset.data_infos[warm_idx]
                        prefix_output_name = (
                            f"{warm_info['token']}__{scene_name}__inst{inst_id:03d}__win{win_id:04d}"
                            f"__history{hidx:02d}.json"
                        )
                        prefix_skip_save = False
                    run_single_sample(
                        model=model,
                        dataset=dataset,
                        index=warm_idx,
                        instruction='',
                        output_name=prefix_output_name,
                        skip_save=prefix_skip_save,
                    )

                info = dataset.data_infos[target_index]
                sample_token = info['token']
                frame_idx = int(info.get('frame_idx', target_pos))
                output_name = (
                    f'{sample_token}__{scene_name}__inst{inst_id:03d}__win{win_id:04d}'
                    f'__frame{frame_idx:04d}.json'
                )
                output_path = osp.join(save_dir, output_name)

                _ = run_single_sample(
                        model=model,
                        dataset=dataset,
                        index=target_index,
                        instruction='' if args.no_instruction else instruction_text,
                        output_name=output_name,
                        skip_save=bool(args.single_window_output_json),
                    )
                if args.single_window_output_json:
                    pred_traj, err, text_out = parse_prediction_from_model_output(_)
                else:
                    pred_traj, err, text_out = parse_prediction_from_file(output_path)

                gt_traj = info['gt_planning'][0, :, :2]
                gt_mask = info['gt_planning_mask'][0]
                fut_valid_flag = bool(gt_mask.all())

                record = dict(
                    scene_name=scene_name,
                    instruction_id=inst_id,
                    instruction=instruction_text,
                    instruction_type=instruction_type,
                    window_id=win_id,
                    sample_token=sample_token,
                    frame_idx=frame_idx,
                    history_indices=[int(dataset.data_infos[i].get('frame_idx', i)) for i in history_indices],
                    output_name=output_name,
                    prediction_error=err,
                    valid_for_metric=False,
                )
                if text_out is not None:
                    record['text_out'] = text_out

                if pred_traj is not None:
                    record['pred_traj_xy'] = pred_traj.tolist()
                record['gt_traj_xy'] = gt_traj[:, :2].tolist()
                record['gt_mask'] = gt_mask.tolist()

                if not fut_valid_flag:
                    record['prediction_error'] = record['prediction_error'] or 'gt_mask_invalid'
                    inst_error_counts['gt_mask_invalid'] += 1
                    inst_records.append(record)
                    log_window_result(rank, record)
                    continue

                if pred_traj is None:
                    inst_error_counts[err] += 1
                    inst_records.append(record)
                    log_window_result(rank, record)
                    continue

                if pred_traj.shape[0] < args.planning_steps:
                    record['prediction_error'] = 'prediction_too_short'
                    inst_error_counts['prediction_too_short'] += 1
                    inst_records.append(record)
                    log_window_result(rank, record)
                    continue

                record.update(dict(valid_for_metric=True))
                if args.compute_ade_per_window:
                    metrics = compute_ades(pred_traj, gt_traj, args.planning_steps)
                    if metrics is None:
                        record['prediction_error'] = 'prediction_too_short'
                        record['valid_for_metric'] = False
                        inst_error_counts['prediction_too_short'] += 1
                    else:
                        record.update(metrics)
                inst_records.append(record)
                log_window_result(rank, record)

            records.extend(inst_records)
            for k, v in inst_error_counts.items():
                local_errors[k] += v
            if args.per_instruction_json_dir:
                safe_type = sanitize_instruction_text(instruction_type, max_len=16)
                safe_hint = sanitize_instruction_text(instruction_text)
                inst_path = osp.join(
                    args.per_instruction_json_dir,
                    f'{scene_name}__inst{inst_id:03d}__type-{safe_type}__{safe_hint}.json',
                )
                inst_payload = dict(
                    scene_name=scene_name,
                    instruction_id=inst_id,
                    instruction=instruction_text,
                    instruction_type=instruction_type,
                    num_windows=len(inst_records),
                    records=inst_records,
                    error_stats=dict(inst_error_counts),
                )
                mmcv.dump(inst_payload, inst_path)
                print(
                    f'[SlidingEval][rank{rank}] saved instruction json: {inst_path} '
                    f'(windows={len(inst_records)})',
                    flush=True,
                )

    if args.per_instruction_json_dir and dist.is_initialized():
        dist.barrier()

    if args.per_instruction_json_dir:
        if not is_main_process(rank):
            if dist.is_initialized():
                dist.destroy_process_group()
            return
        all_records, inst_json_paths = load_records_from_instruction_json_dir(args.per_instruction_json_dir)
        print(f'[SlidingEval] loaded instruction json files: {len(inst_json_paths)}', flush=True)
        global_errors = defaultdict(int)
        for r in all_records:
            if not r.get('valid_for_metric', False):
                k = r.get('prediction_error', 'unknown')
                global_errors[k] += 1
        global_errors = dict(global_errors)
    else:
        all_records = gather_records(records, rank, world_size)
        global_errors = gather_error_stats(local_errors, rank, world_size)
    if not is_main_process(rank):
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    extra_errors = finalize_ade_for_records(all_records, args.planning_steps)
    for k, v in extra_errors.items():
        global_errors[k] = global_errors.get(k, 0) + v

    global_summary = summarize_metric_records(all_records, args.planning_steps)
    global_error_summary = summarize_error_records(all_records)

    per_scene_summary = {}
    for scene_name in sorted({r['scene_name'] for r in all_records}):
        scene_records = [r for r in all_records if r['scene_name'] == scene_name]
        per_scene_summary[scene_name] = summarize_metric_records(scene_records, args.planning_steps)

    instruction_type_summary = {}
    instruction_type_error_stats = {}
    instruction_type_error_summary = {}
    instruction_types = sorted({r.get('instruction_type', 'unknown') for r in all_records})
    for inst_type in instruction_types:
        type_records = [r for r in all_records if r.get('instruction_type', 'unknown') == inst_type]
        type_summary = summarize_metric_records(type_records, args.planning_steps)
        type_summary['total_records'] = len(type_records)
        instruction_type_summary[inst_type] = type_summary

        type_errors = defaultdict(int)
        for r in type_records:
            if not r.get('valid_for_metric', False):
                k = r.get('prediction_error', 'unknown')
                type_errors[k] += 1
        instruction_type_error_stats[inst_type] = dict(type_errors)
        instruction_type_error_summary[inst_type] = summarize_error_records(type_records)

    print('[SlidingEval] ===== Final Metrics =====')
    print(f"[SlidingEval] valid_windows={global_summary['count']} / total_records={len(all_records)}")
    for key in metric_keys(args.planning_steps):
        print(f"[SlidingEval] {key.upper()}={global_summary[key]}")
    print(f"[SlidingEval] MADE={(global_summary['made'])}")
    print(
        f"[SlidingEval] error_windows={global_error_summary['error_count']} / "
        f"total_records={global_error_summary['total_records']} "
        f"(error_rate={global_error_summary['error_rate']:.4f})"
    )

    print('[SlidingEval] instruction_type_metrics:')
    for inst_type in instruction_types:
        s = instruction_type_summary[inst_type]
        es = instruction_type_error_summary[inst_type]
        print(
            f"  - {inst_type}: valid={s['count']} / total={s['total_records']}, "
            f"errors={es['error_count']} (error_rate={es['error_rate']:.4f}), "
            f"{format_metrics(s)}"
        )

    if global_errors:
        print('[SlidingEval] error_stats:')
        for k in sorted(global_errors.keys()):
            print(f'  - {k}: {global_errors[k]}')

    output_payload = dict(
        config=args.config,
        checkpoint=args.checkpoint,
        doscenes_csv=args.doscenes_csv,
        save_dir=save_dir,
        world_size=world_size,
        fps=args.fps,
        history_seconds=args.history_seconds,
        future_seconds=args.future_seconds,
        planning_steps=args.planning_steps,
        history_frames=history_frames,
        future_frames=future_frames,
        window_stride=args.window_stride,
        global_metrics=global_summary,
        global_error_summary=global_error_summary,
        scene_metrics=per_scene_summary,
        instruction_type_metrics=instruction_type_summary,
        instruction_type_error_stats=instruction_type_error_stats,
        instruction_type_error_summary=instruction_type_error_summary,
        total_records=len(all_records),
        error_stats=dict(global_errors),
        per_instruction_json_dir=args.per_instruction_json_dir,
    )

    if args.metrics_json:
        mmcv.dump(output_payload, args.metrics_json)
        print(f'[SlidingEval] metrics saved to {args.metrics_json}')

    if args.pred_record_json:
        mmcv.dump(all_records, args.pred_record_json)
        print(f'[SlidingEval] prediction records saved to {args.pred_record_json}')

    if args.single_window_output_json:
        mmcv.dump(all_records, args.single_window_output_json)
        print(f'[SlidingEval] single window output saved to {args.single_window_output_json}')

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
