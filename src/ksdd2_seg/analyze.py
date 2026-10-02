from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .metrics import (
    box_iou,
    dice,
    greedy_match,
    mask_iou,
    metric_matrix,
    read_label_instances,
    read_prediction_instances,
    safe_divide,
)


def _classify_false_positives(
    box_matrix: np.ndarray,
    box_matches: dict[int, int],
    prediction_count: int,
    duplicate_iou: float,
) -> Counter[str]:
    matched_predictions = set(box_matches.values())
    counts: Counter[str] = Counter()
    for prediction_index in range(prediction_count):
        if prediction_index in matched_predictions:
            continue
        if box_matrix.shape[0] == 0:
            counts["normal_image"] += 1
            continue
        best_gt = int(np.argmax(box_matrix[:, prediction_index]))
        best_iou = float(box_matrix[best_gt, prediction_index])
        if best_iou >= duplicate_iou and best_gt in box_matches:
            counts["duplicate"] += 1
        elif best_iou >= duplicate_iou:
            counts["localization"] += 1
        else:
            counts["off_target"] += 1
    return counts


def _classify_false_negatives(
    box_matrix: np.ndarray,
    box_matches: dict[int, int],
    ground_truth_count: int,
    prediction_count: int,
    match_iou: float,
    duplicate_iou: float,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for gt_index in range(ground_truth_count):
        if gt_index in box_matches:
            continue
        if prediction_count == 0:
            counts["no_prediction"] += 1
            continue
        best_iou = float(box_matrix[gt_index].max())
        if best_iou >= match_iou:
            counts["assignment_conflict"] += 1
        elif best_iou >= duplicate_iou:
            counts["localization"] += 1
        else:
            counts["weak_overlap"] += 1
    return counts


def evaluate_instances(
    ground_truth: list[Any],
    predictions: list[Any],
    match_iou: float,
    duplicate_iou: float,
) -> tuple[dict[str, Any], dict[int, int], dict[int, int]]:
    """Evaluate one image with the same one-to-one box/mask matching protocol as the report."""
    box_matrix = metric_matrix(ground_truth, predictions, box_iou, "box")
    mask_matrix = metric_matrix(ground_truth, predictions, mask_iou, "mask")
    box_matches = greedy_match(box_matrix, match_iou)
    mask_matches = greedy_match(mask_matrix, match_iou)
    fp_types = _classify_false_positives(box_matrix, box_matches, len(predictions), duplicate_iou)
    fn_types = _classify_false_negatives(
        box_matrix,
        box_matches,
        len(ground_truth),
        len(predictions),
        match_iou,
        duplicate_iou,
    )

    paired_mask_ious = [
        mask_iou(ground_truth[gt_index].mask, predictions[pred_index].mask)
        for gt_index, pred_index in box_matches.items()
    ]
    paired_dices = [
        dice(ground_truth[gt_index].mask, predictions[pred_index].mask)
        for gt_index, pred_index in box_matches.items()
    ]
    row = {
        "gt_count": len(ground_truth),
        "pred_count": len(predictions),
        "box_matched": len(box_matches),
        "box_fn": len(ground_truth) - len(box_matches),
        "box_fp": len(predictions) - len(box_matches),
        "mask_matched": len(mask_matches),
        "mask_fn": len(ground_truth) - len(mask_matches),
        "mask_fp": len(predictions) - len(mask_matches),
        "normal_image_fp": fp_types["normal_image"],
        "duplicate_fp": fp_types["duplicate"],
        "localization_fp": fp_types["localization"],
        "off_target_fp": fp_types["off_target"],
        "fn_no_prediction": fn_types["no_prediction"],
        "fn_weak_overlap": fn_types["weak_overlap"],
        "fn_localization": fn_types["localization"],
        "fn_assignment_conflict": fn_types["assignment_conflict"],
        "paired_mask_iou_sum": sum(paired_mask_ious),
        "paired_dice_sum": sum(paired_dices),
    }
    return row, box_matches, mask_matches


def summarize(image_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def total(key: str) -> float:
        return sum(float(row[key]) for row in image_rows)

    images = len(image_rows)
    positive_images = sum(int(row["gt_count"] > 0) for row in image_rows)
    normal_images = images - positive_images
    localized_images = sum(int(row["gt_count"] > 0 and row["box_matched"] > 0) for row in image_rows)
    normal_alarm_images = sum(int(row["gt_count"] == 0 and row["pred_count"] > 0) for row in image_rows)
    box_matched = total("box_matched")
    box_predictions = total("pred_count")
    box_ground_truth = total("gt_count")
    mask_matched = total("mask_matched")
    box_precision = safe_divide(box_matched, box_predictions)
    box_recall = safe_divide(box_matched, box_ground_truth)
    mask_precision = safe_divide(mask_matched, box_predictions)
    mask_recall = safe_divide(mask_matched, box_ground_truth)

    return {
        "images": images,
        "positive_images": positive_images,
        "normal_images": normal_images,
        "gt_instances": int(box_ground_truth),
        "pred_instances": int(box_predictions),
        "box_matched": int(box_matched),
        "box_fn": int(box_ground_truth - box_matched),
        "box_fp": int(box_predictions - box_matched),
        "box_precision": box_precision,
        "box_recall": box_recall,
        "box_f1": safe_divide(2 * box_precision * box_recall, box_precision + box_recall),
        "mask_matched": int(mask_matched),
        "mask_fn": int(box_ground_truth - mask_matched),
        "mask_fp": int(box_predictions - mask_matched),
        "mask_precision": mask_precision,
        "mask_recall": mask_recall,
        "mask_f1": safe_divide(2 * mask_precision * mask_recall, mask_precision + mask_recall),
        "localized_positive_images": localized_images,
        "image_localization_recall": safe_divide(localized_images, positive_images),
        "normal_false_alarm_images": normal_alarm_images,
        "normal_image_false_alarm_rate": safe_divide(normal_alarm_images, normal_images),
        "mean_mask_iou_for_box_matches": safe_divide(total("paired_mask_iou_sum"), box_matched),
        "mean_dice_for_box_matches": safe_divide(total("paired_dice_sum"), box_matched),
        "false_positive_types": {
            "duplicate": int(total("duplicate_fp")),
            "localization": int(total("localization_fp")),
            "off_target": int(total("off_target_fp")),
            "normal_image": int(total("normal_image_fp")),
        },
        "false_negative_types": {
            "no_prediction": int(total("fn_no_prediction")),
            "weak_overlap": int(total("fn_weak_overlap")),
            "localization": int(total("fn_localization")),
            "assignment_conflict": int(total("fn_assignment_conflict")),
        },
    }


def _preview_categories(row: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    if row["gt_count"] > 0 and row["box_fn"] == 0 and row["box_fp"] == 0:
        categories.append("good_case")
    if row["box_fn"]:
        categories.append("false_negative")
    if row["box_fp"]:
        categories.append("false_positive")
    if row["gt_count"] == 0 and row["pred_count"]:
        categories.append("normal_image_fp")
    if row["mask_fn"] and not row["box_fn"]:
        categories.append("mask_error")
    return categories


def _write_preview(result: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotted_bgr = result.plot()
    Image.fromarray(plotted_bgr[..., ::-1]).save(output_path, quality=90)


def write_outputs(
    output_dir: Path,
    image_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    settings: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(image_rows[0].keys()) if image_rows else ["image_name"]
    with (output_dir / "image_results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(image_rows)
    payload = {"settings": settings, "summary": summary}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Operating-point error analysis",
        "",
        f"- Split: `{settings['split']}`",
        f"- Confidence: `{settings['confidence']:.4f}`",
        f"- NMS IoU: `{settings['nms_iou']}`",
        f"- Box/mask match IoU: `{settings['match_iou']}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| Box precision | {summary['box_precision']:.3f} |",
        f"| Box recall | {summary['box_recall']:.3f} |",
        f"| Mask precision | {summary['mask_precision']:.3f} |",
        f"| Mask recall | {summary['mask_recall']:.3f} |",
        (
            f"| Localized positive images | {summary['localized_positive_images']} / "
            f"{summary['positive_images']} ({summary['image_localization_recall']:.1%}) |"
        ),
        (
            f"| Normal-image false alarms | {summary['normal_false_alarm_images']} / "
            f"{summary['normal_images']} ({summary['normal_image_false_alarm_rate']:.2%}) |"
        ),
        f"| Mean mask IoU for box matches | {summary['mean_mask_iou_for_box_matches']:.3f} |",
        f"| Mean Dice for box matches | {summary['mean_dice_for_box_matches']:.3f} |",
        "",
        "Generated previews are local artifacts and are ignored by Git.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a frozen YOLO segmentation operating point.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.0975)
    parser.add_argument("--iou", type=float, default=0.3, help="Prediction NMS IoU.")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--duplicate-iou", type=float, default=0.3)
    parser.add_argument("--device")
    parser.add_argument("--max-previews", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    images_dir = args.dataset / "images" / args.split
    labels_dir = args.dataset / "labels" / args.split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing prepared split: {images_dir}")

    from ultralytics import YOLO

    predict_args: dict[str, Any] = {
        "source": str(images_dir.resolve()),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "save": False,
        "verbose": False,
        "stream": True,
    }
    if args.device is not None:
        predict_args["device"] = args.device

    image_rows: list[dict[str, Any]] = []
    preview_counts: Counter[str] = Counter()
    for result in YOLO(args.model).predict(**predict_args):
        image_path = Path(result.path)
        height, width = result.orig_shape
        ground_truth = read_label_instances(labels_dir / f"{image_path.stem}.txt", width, height)
        predictions = read_prediction_instances(result, width, height)
        row, _box_matches, _mask_matches = evaluate_instances(
            ground_truth, predictions, args.match_iou, args.duplicate_iou
        )
        row = {"image_name": image_path.name, **row}
        image_rows.append(row)
        for category in _preview_categories(row):
            if preview_counts[category] < args.max_previews:
                _write_preview(result, args.output / "previews" / category / f"{image_path.stem}.jpg")
                preview_counts[category] += 1

    image_rows.sort(key=lambda row: row["image_name"])
    summary = summarize(image_rows)
    settings = {
        "model": Path(args.model).name,
        "dataset": args.dataset.name,
        "split": args.split,
        "imgsz": args.imgsz,
        "confidence": args.conf,
        "nms_iou": args.iou,
        "match_iou": args.match_iou,
        "duplicate_iou": args.duplicate_iou,
    }
    write_outputs(args.output, image_rows, summary, settings)
    print(json.dumps({"settings": settings, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
