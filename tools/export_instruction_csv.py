import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export per-instruction evaluation results to CSV."
    )
    parser.add_argument(
        "input_json",
        help=(
            "Path to evaluation result input. Supports sliding-window all_windows.json, "
            "full-scene index.json, or a full-scene instruction json directory."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        default=None,
        help="Output CSV path. Defaults to <input_dir>/vlm_doscenes_results.csv",
    )
    return parser.parse_args()


def scene_sort_key(scene_name):
    match = re.search(r"(\d+)$", scene_name or "")
    if match:
        return int(match.group(1))
    return float("inf")


def scene_number(scene_name):
    key = scene_sort_key(scene_name)
    if key == float("inf"):
        return scene_name or ""
    return str(key)


def mean_or_blank(total, count):
    if count <= 0:
        return ""
    return total / count


def compute_ades_from_record(record):
    pred = record.get("pred_traj_xy") or []
    gt = record.get("gt_traj_xy") or []
    if len(pred) < 6 or len(gt) < 6:
        return None

    dists = []
    for i in range(6):
        try:
            px, py = float(pred[i][0]), float(pred[i][1])
            gx, gy = float(gt[i][0]), float(gt[i][1])
        except (TypeError, ValueError, IndexError):
            return None
        dists.append(math.sqrt((px - gx) ** 2 + (py - gy) ** 2))

    ade1 = sum(dists[:2]) / 2.0
    ade2 = sum(dists[:4]) / 4.0
    ade3 = sum(dists[:6]) / 6.0
    made = (ade1 + ade2 + ade3) / 3.0
    return ade1, ade2, ade3, made


def load_sliding_window_records(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    per_instruction = defaultdict(
        lambda: {
            "scene_name": "",
            "instruction_type": "",
            "instruction": "",
            "count": 0,
            "ade1s_sum": 0.0,
            "ade2s_sum": 0.0,
            "ade3s_sum": 0.0,
            "made_sum": 0.0,
        }
    )

    for record in records:
        key = (
            record.get("scene_name", ""),
            int(record.get("instruction_id", -1)),
            record.get("instruction_type", "") or "",
            record.get("instruction", "") or "",
        )
        row = per_instruction[key]
        row["scene_name"] = key[0]
        row["instruction_type"] = key[2]
        row["instruction"] = key[3]

        if not record.get("valid_for_metric"):
            continue

        row["count"] += 1
        row["ade1s_sum"] += float(record.get("ade1s", 0.0))
        row["ade2s_sum"] += float(record.get("ade2s", 0.0))
        row["ade3s_sum"] += float(record.get("ade3s", 0.0))
        row["made_sum"] += float(record.get("made", 0.0))

    return per_instruction


def load_full_scene_instruction_dir(input_dir):
    per_instruction = {}
    for path in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        key = (
            payload.get("scene_name", ""),
            int(payload.get("instruction_id", -1)),
            payload.get("instruction_type", "") or "",
            payload.get("instruction", "") or "",
        )
        summary = payload.get("metric_summary") or {}
        per_instruction[key] = {
            "scene_name": key[0],
            "instruction_type": key[2],
            "instruction": key[3],
            "count": int(summary.get("count") or 0),
            "ade1s_sum": float(summary["ade1s"]) * int(summary.get("count") or 0)
            if summary.get("ade1s") is not None
            else 0.0,
            "ade2s_sum": float(summary["ade2s"]) * int(summary.get("count") or 0)
            if summary.get("ade2s") is not None
            else 0.0,
            "ade3s_sum": float(summary["ade3s"]) * int(summary.get("count") or 0)
            if summary.get("ade3s") is not None
            else 0.0,
            "made_sum": float(summary["made"]) * int(summary.get("count") or 0)
            if summary.get("made") is not None
            else 0.0,
        }

    return per_instruction


def load_instruction_dir_with_records(input_dir):
    per_instruction = {}
    for path in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        key = (
            payload.get("scene_name", ""),
            int(payload.get("instruction_id", -1)),
            payload.get("instruction_type", "") or "",
            payload.get("instruction", "") or "",
        )
        row = {
            "scene_name": key[0],
            "instruction_type": key[2],
            "instruction": key[3],
            "count": 0,
            "ade1s_sum": 0.0,
            "ade2s_sum": 0.0,
            "ade3s_sum": 0.0,
            "made_sum": 0.0,
        }

        records = payload.get("records") or []
        for record in records:
            if not record.get("valid_for_metric"):
                continue

            ades = None
            if all(record.get(k) is not None for k in ["ade1s", "ade2s", "ade3s", "made"]):
                ades = (
                    float(record["ade1s"]),
                    float(record["ade2s"]),
                    float(record["ade3s"]),
                    float(record["made"]),
                )
            else:
                ades = compute_ades_from_record(record)

            if ades is None:
                continue

            ade1, ade2, ade3, made = ades
            row["count"] += 1
            row["ade1s_sum"] += ade1
            row["ade2s_sum"] += ade2
            row["ade3s_sum"] += ade3
            row["made_sum"] += made

        per_instruction[key] = row
    return per_instruction


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input_json)

    if os.path.isdir(input_path):
        per_instruction = load_instruction_dir_with_records(input_path)
    elif input_path.endswith(".instruction_jsons"):
        per_instruction = load_instruction_dir_with_records(input_path)
    else:
        per_instruction = load_sliding_window_records(input_path)

    sorted_rows = sorted(
        per_instruction.items(),
        key=lambda item: (scene_sort_key(item[0][0]), item[0][1], item[0][3]),
    )

    output_csv = args.output_csv
    if not output_csv:
        base_dir = input_path if os.path.isdir(input_path) else os.path.dirname(input_path)
        output_csv = os.path.join(base_dir, "vlm_doscenes_results.csv")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test",
                "instruction_type",
                "instruction",
                "name",
                "ade1s",
                "ade2s",
                "ade3s",
                "ade",
            ],
        )
        writer.writeheader()
        for (_, _, _, _), row in sorted_rows:
            writer.writerow(
                {
                    "test": scene_number(row["scene_name"]),
                    "instruction_type": row["instruction_type"],
                    "instruction": row["instruction"],
                    "name": scene_number(row["scene_name"]),
                    "ade1s": mean_or_blank(row["ade1s_sum"], row["count"]),
                    "ade2s": mean_or_blank(row["ade2s_sum"], row["count"]),
                    "ade3s": mean_or_blank(row["ade3s_sum"], row["count"]),
                    "ade": mean_or_blank(row["made_sum"], row["count"]),
                }
            )

    print(f"saved {len(sorted_rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
