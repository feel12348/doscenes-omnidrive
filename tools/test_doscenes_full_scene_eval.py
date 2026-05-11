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

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description='doScenes full-scene sequential evaluation (single GPU)')
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
        '--index-json',
        default=None,
        help='optional json path to dump output metadata index')
    parser.add_argument(
        '--per-instruction-json-dir',
        default=None,
        help='optional directory to dump one json per instruction')
    parser.add_argument(
        '--metrics-json',
        default=None,
        help='optional path to dump final metrics json')
    parser.add_argument(
        '--max-scenes',
        type=int,
        default=0,
        help='debug option: only process first N scenes (0 means all)')
    parser.add_argument(
        '--max-frames-per-scene',
        type=int,
        default=0,
        help='debug option: only process first N frames per scene (0 means all)')
    parser.add_argument(
        '--frame-stride',
        type=int,
        default=1,
        help='evaluate one frame every N frames in each scene (>=1)')
    parser.add_argument(
        '--fps',
        type=float,
        default=2.0,
        help='frame rate used to convert history-seconds to warm-up frames')
    parser.add_argument(
        '--history-seconds',
        type=float,
        default=2.0,
        help='warm-up seconds before applying instruction')
    parser.add_argument(
        '--history-frames',
        type=int,
        default=0,
        help='if >0, overrides history-seconds')
    parser.add_argument(
        '--planning-steps',
        type=int,
        default=6,
        help='number of future trajectory waypoints required for metrics; use 12 for 6s at 2Hz')
    parser.add_argument(
        '--drop-tail-frames',
        type=int,
        default=0,
        help='drop last N frames of each scene from sequential evaluation')
    parser.add_argument(
        '--record-warmup-outputs',
        action='store_true',
        help='if set, save outputs of warm-up frames; default skip saving them')
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


def gather_int(value, rank, world_size):
    if world_size == 1:
        return value
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, int(value))
    if rank == 0:
        return int(sum(gathered))
    return None


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
            if isinstance(a, list) and len(a) > 0:
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


def sanitize_instruction_text(text, max_len=48):
    s = re.sub(r'[^0-9a-zA-Z_-]+', '_', (text or '').strip())
    s = re.sub(r'_+', '_', s).strip('_')
    if not s:
        s = 'empty'
    return s[:max_len]


def build_scene_indices(dataset):
    scene_to_indices = defaultdict(list)
    for idx, info in enumerate(dataset.data_infos):
        scene_to_indices[info.get('scene_name', '')].append(idx)
    for scene_name, indices in scene_to_indices.items():
        indices.sort(key=lambda i: dataset.data_infos[i].get('frame_idx', i))
        scene_to_indices[scene_name] = indices
    return scene_to_indices


def get_history_frames(args):
    if args.history_frames > 0:
        return args.history_frames
    return max(int(round(args.history_seconds * args.fps)), 0)


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed(args)
    cfg = prepare_cfg(args)
    model, dataset = build_model_and_dataset(
        cfg, args.checkpoint, save_dir=args.save_dir, device_id=local_rank)

    if args.save_dir is None:
        save_dir = model.module.save_path
    else:
        save_dir = args.save_dir
    mmcv.mkdir_or_exist(save_dir)
    per_instruction_json_dir = args.per_instruction_json_dir
    if per_instruction_json_dir is None and args.index_json:
        per_instruction_json_dir = args.index_json + '.instruction_jsons'
    if per_instruction_json_dir is not None:
        mmcv.mkdir_or_exist(per_instruction_json_dir)

    scene_to_insts = load_instruction_map(args.doscenes_csv)
    scene_to_indices = build_scene_indices(dataset)
    history_frames = get_history_frames(args)

    selected_scenes = [s for s in scene_to_indices.keys() if s in scene_to_insts]
    selected_scenes.sort()
    if args.max_scenes > 0:
        selected_scenes = selected_scenes[:args.max_scenes]

    metadata_index = []
    local_total_outputs = 0
    if is_main_process(rank):
        print(
            f'[FullSceneEval] scenes={len(selected_scenes)}, '
            f'history_frames={history_frames}, frame_stride={args.frame_stride}, '
            f'world_size={world_size}'
        )

    instruction_task_id = 0
    for scene_pos, scene_name in enumerate(selected_scenes):
        frame_indices = scene_to_indices[scene_name]
        if args.max_frames_per_scene > 0:
            frame_indices = frame_indices[:args.max_frames_per_scene]
        if args.frame_stride > 1:
            frame_indices = frame_indices[::args.frame_stride]
        instructions = scene_to_insts[scene_name]

        if is_main_process(rank):
            print(
                f'[FullSceneEval] scene {scene_pos + 1}/{len(selected_scenes)} '
                f'{scene_name}: frames={len(frame_indices)}, instructions={len(instructions)}'
            )

        for inst_id, instruction_item in enumerate(instructions):
            instruction_text = (
                instruction_item.get('instruction', '')
                if isinstance(instruction_item, dict) else str(instruction_item)
            )
            instruction_type = (
                instruction_item.get('instruction_type', 'unknown')
                if isinstance(instruction_item, dict) else 'unknown'
            )
            process_this_instruction = (instruction_task_id % world_size == rank)
            instruction_task_id += 1
            if not process_this_instruction:
                continue

            reset_temporal_memory(model)
            inst_records = []

            # Warm-up phase: advance temporal memory without instruction.
            warm_end = min(history_frames, len(frame_indices))
            for warm_pos in range(warm_end):
                warm_idx = frame_indices[warm_pos]
                info = dataset.data_infos[warm_idx]
                sample_token = info['token']
                frame_idx = int(info.get('frame_idx', warm_pos))
                output_name = (
                    f'{sample_token}__{scene_name}__inst{inst_id:03d}'
                    f'__warmup__frame{frame_idx:04d}.json'
                )
                _ = run_single_sample(
                    model=model,
                    dataset=dataset,
                    index=warm_idx,
                    instruction='',
                    output_name=output_name if args.record_warmup_outputs else None,
                    skip_save=(not args.record_warmup_outputs),
                )

            # Sequential full-scene phase: apply instruction from warm_end to
            # end_pos (optionally dropping tail frames for fair comparison).
            end_pos = len(frame_indices) - max(args.drop_tail_frames, 0)
            if end_pos <= warm_end:
                continue
            for pos in range(warm_end, end_pos):
                idx = frame_indices[pos]
                info = dataset.data_infos[idx]
                sample_token = info['token']
                frame_idx = int(info.get('frame_idx', pos))
                instruction = '' if args.no_instruction else instruction_text
                output_name = (
                    f'{sample_token}__{scene_name}__inst{inst_id:03d}'
                    f'__seq__frame{frame_idx:04d}.json'
                )
                outputs = run_single_sample(
                    model=model,
                    dataset=dataset,
                    index=idx,
                    instruction=instruction,
                    output_name=output_name,
                    skip_save=False,
                )
                output_path = osp.join(save_dir, output_name)
                pred_traj, pred_err, text_out = parse_prediction_from_file(output_path)
                if pred_err == 'prediction_file_missing':
                    pred_traj, pred_err, text_out = parse_prediction_from_model_output(outputs)

                gt_traj = info['gt_planning'][0, :, :2]
                gt_mask = info['gt_planning_mask'][0]
                fut_valid_flag = bool(gt_mask.all())

                record = dict(
                    output_name=output_name,
                    sample_token=sample_token,
                    scene_name=scene_name,
                    frame_idx=frame_idx,
                    instruction_id=inst_id,
                    instruction_type=instruction_type,
                    instruction=instruction_text,
                    applied_instruction=(instruction != ''),
                    phase='sequential',
                    prediction_error=pred_err,
                    valid_output=(pred_err is None),
                    valid_for_metric=False,
                    text_out=text_out,
                    gt_traj_xy=gt_traj[:, :2].tolist(),
                    gt_mask=gt_mask.tolist(),
                )
                if pred_traj is not None:
                    record['pred_traj_xy'] = pred_traj.tolist()

                if not fut_valid_flag:
                    record['prediction_error'] = record['prediction_error'] or 'gt_mask_invalid'
                elif pred_traj is None:
                    pass
                elif pred_traj.shape[0] < args.planning_steps:
                    record['prediction_error'] = 'prediction_too_short'
                else:
                    metrics = compute_ades(pred_traj, gt_traj, args.planning_steps)
                    if metrics is None:
                        record['prediction_error'] = 'prediction_too_short'
                    else:
                        record['valid_for_metric'] = True
                        record.update(metrics)

                metadata_index.append(record)
                inst_records.append(metadata_index[-1])
                local_total_outputs += 1
                if local_total_outputs % 100 == 0:
                    print(f'[FullSceneEval][rank{rank}] completed outputs: {local_total_outputs}')

            if per_instruction_json_dir is not None and inst_records:
                safe_type = sanitize_instruction_text(instruction_type, max_len=16)
                safe_hint = sanitize_instruction_text(instruction_text)
                inst_path = osp.join(
                    per_instruction_json_dir,
                    f'{scene_name}__inst{inst_id:03d}__type-{safe_type}__{safe_hint}.json',
                )
                inst_payload = dict(
                    scene_name=scene_name,
                    instruction_id=inst_id,
                    instruction=instruction_text,
                    instruction_type=instruction_type,
                    num_records=len(inst_records),
                    records=inst_records,
                    metric_summary=summarize_metric_records(inst_records, args.planning_steps),
                    error_summary=summarize_error_records(inst_records),
                )
                mmcv.dump(inst_payload, inst_path)
                print(
                    f'[FullSceneEval][rank{rank}] saved instruction json: {inst_path} '
                    f'(records={len(inst_records)})',
                    flush=True,
                )

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
    all_records = gather_records(metadata_index, rank, world_size)
    total_outputs = gather_int(local_total_outputs, rank, world_size)

    if is_main_process(rank):
        extra_errors = finalize_ade_for_records(all_records, args.planning_steps)
        global_error_summary = summarize_error_records(all_records)
        global_summary = summarize_metric_records(all_records, args.planning_steps)
        global_errors = dict(global_error_summary['error_stats'])
        for k, v in extra_errors.items():
            global_errors[k] = global_errors.get(k, 0) + v
        global_error_summary['error_stats'] = global_errors
        global_error_summary['error_count'] = int(sum(global_errors.values()))
        if global_error_summary['total_records'] > 0:
            global_error_summary['error_rate'] = (
                float(global_error_summary['error_count']) /
                float(global_error_summary['total_records'])
            )
        instruction_type_error_summary = {}
        instruction_type_metrics = {}
        instruction_types = sorted({r.get('instruction_type', 'unknown') for r in all_records})
        for inst_type in instruction_types:
            type_records = [r for r in all_records if r.get('instruction_type', 'unknown') == inst_type]
            instruction_type_error_summary[inst_type] = summarize_error_records(type_records)
            type_summary = summarize_metric_records(type_records, args.planning_steps)
            type_summary['total_records'] = len(type_records)
            instruction_type_metrics[inst_type] = type_summary

        per_scene_metrics = {}
        for scene_name in sorted({r['scene_name'] for r in all_records}):
            scene_records = [r for r in all_records if r['scene_name'] == scene_name]
            per_scene_metrics[scene_name] = summarize_metric_records(scene_records, args.planning_steps)

        print(f'[FullSceneEval] done. total outputs={total_outputs}, save_dir={save_dir}')
        print('[FullSceneEval] ===== Final Metrics =====')
        print(f"[FullSceneEval] valid_records={global_summary['count']} / total_records={len(all_records)}")
        for key in metric_keys(args.planning_steps):
            print(f"[FullSceneEval] {key.upper()}={global_summary[key]}")
        print(f"[FullSceneEval] MADE={global_summary['made']}")
        print(
            f"[FullSceneEval] error_outputs={global_error_summary['error_count']} / "
            f"total_records={global_error_summary['total_records']} "
            f"(error_rate={global_error_summary['error_rate']:.4f})"
        )
        if global_error_summary['error_stats']:
            print('[FullSceneEval] error_stats:')
            for k in sorted(global_error_summary['error_stats'].keys()):
                print(f"  - {k}: {global_error_summary['error_stats'][k]}")
        print('[FullSceneEval] instruction_type_metrics:')
        for inst_type in instruction_types:
            m = instruction_type_metrics[inst_type]
            s = instruction_type_error_summary[inst_type]
            print(
                f"  - {inst_type}: valid={m['count']} / total={m['total_records']}, "
                f"errors={s['error_count']} (error_rate={s['error_rate']:.4f}), "
                f"{format_metrics(m)}"
            )
        if args.metrics_json:
            metrics_payload = dict(
                config=args.config,
                checkpoint=args.checkpoint,
                doscenes_csv=args.doscenes_csv,
                save_dir=save_dir,
                total_outputs=total_outputs,
                planning_steps=args.planning_steps,
                global_metrics=global_summary,
                global_error_summary=global_error_summary,
                scene_metrics=per_scene_metrics,
                instruction_type_metrics=instruction_type_metrics,
                instruction_type_error_summary=instruction_type_error_summary,
                per_instruction_json_dir=per_instruction_json_dir,
            )
            mmcv.dump(metrics_payload, args.metrics_json)
            print(f'[FullSceneEval] metrics saved to {args.metrics_json}')
        if args.index_json:
            mmcv.dump(all_records, args.index_json)
            print(f'[FullSceneEval] index saved to {args.index_json}')
            summary_path = args.index_json + '.summary.json'
            summary_payload = dict(
                config=args.config,
                checkpoint=args.checkpoint,
                doscenes_csv=args.doscenes_csv,
                save_dir=save_dir,
                total_outputs=total_outputs,
                planning_steps=args.planning_steps,
                global_metrics=global_summary,
                global_error_summary=global_error_summary,
                scene_metrics=per_scene_metrics,
                instruction_type_metrics=instruction_type_metrics,
                instruction_type_error_summary=instruction_type_error_summary,
                per_instruction_json_dir=per_instruction_json_dir,
            )
            mmcv.dump(summary_payload, summary_path)
            print(f'[FullSceneEval] summary saved to {summary_path}')


if __name__ == '__main__':
    main()
