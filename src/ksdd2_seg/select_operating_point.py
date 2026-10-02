from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from .analyze import evaluate_instances, summarize
from .metrics import read_label_instances, read_prediction_instances


def confidence_grid(start: float, stop: float, step: float) -> list[float]:
    """Build a decimal confidence grid without accumulating binary-float error."""
    if not 0.0 <= start <= stop <= 1.0:
        raise ValueError("Confidence range must satisfy 0 <= start <= stop <= 1.")
    if step <= 0:
        raise ValueError("Confidence step must be positive.")
    current = Decimal(str(start))
    final = Decimal(str(stop))
    increment = Decimal(str(step))
    values: list[float] = []
    while current <= final:
        values.append(float(current))
        current += increment
    return values


def select_best(rows: list[dict[str, Any]], max_normal_alarms: int) -> dict[str, Any]:
    """Select maximum localization under a fixed normal-image alarm budget."""
    feasible = [row for row in rows if row["normal_false_alarm_images"] <= max_normal_alarms]
    if not feasible:
        raise ValueError("No confidence threshold satisfies the normal-image alarm budget.")
    return max(
        feasible,
        key=lambda row: (
            row["localized_positive_images"],
            -row["box_fn"],
            row["box_f1"],
            row["box_precision"],
            row["confidence"],
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a validation confidence under a fixed normal-image alarm budget."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.3, help="Fixed prediction NMS IoU.")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--duplicate-iou", type=float, default=0.3)
    parser.add_argument("--conf-start", type=float, default=0.02)
    parser.add_argument("--conf-stop", type=float, default=0.30)
    parser.add_argument("--conf-step", type=float, default=0.0025)
    parser.add_argument("--max-normal-alarms", type=int, default=5)
    parser.add_argument("--device")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    images_dir = args.dataset / "images" / args.split
    labels_dir = args.dataset / "labels" / args.split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing prepared validation split: {images_dir}")

    thresholds = confidence_grid(args.conf_start, args.conf_stop, args.conf_step)
    rows_by_threshold: dict[float, list[dict[str, Any]]] = {value: [] for value in thresholds}

    from ultralytics import YOLO

    predict_args: dict[str, Any] = {
        "source": str(images_dir.resolve()),
        "imgsz": args.imgsz,
        "conf": min(thresholds),
        "iou": args.iou,
        "save": False,
        "verbose": False,
        "stream": True,
    }
    if args.device is not None:
        predict_args["device"] = args.device

    for result in YOLO(args.model).predict(**predict_args):
        image_path = Path(result.path)
        height, width = result.orig_shape
        ground_truth = read_label_instances(labels_dir / f"{image_path.stem}.txt", width, height)
        all_predictions = read_prediction_instances(result, width, height)
        for threshold in thresholds:
            predictions = [item for item in all_predictions if item.confidence >= threshold]
            row, _box_matches, _mask_matches = evaluate_instances(
                ground_truth, predictions, args.match_iou, args.duplicate_iou
            )
            rows_by_threshold[threshold].append(row)

    sweep_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        row = summarize(rows_by_threshold[threshold])
        sweep_rows.append({"confidence": threshold, **row})
    selected = select_best(sweep_rows, args.max_normal_alarms)

    args.output.mkdir(parents=True, exist_ok=True)
    scalar_fields = [
        "confidence",
        "box_precision",
        "box_recall",
        "box_f1",
        "mask_precision",
        "mask_recall",
        "mask_f1",
        "localized_positive_images",
        "positive_images",
        "box_fn",
        "box_fp",
        "normal_false_alarm_images",
        "normal_images",
        "normal_image_false_alarm_rate",
    ]
    with (args.output / "confidence_sweep.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=scalar_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sweep_rows)

    payload = {
        "selection_rule": (
            "maximize localized positive images, then minimize box FN, then maximize box F1, "
            "box precision, and confidence, subject to the normal-image alarm budget"
        ),
        "max_normal_alarm_images": args.max_normal_alarms,
        "nms_iou": args.iou,
        "match_iou": args.match_iou,
        "selected": selected,
    }
    (args.output / "selected_operating_point.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (args.output / "selected_operating_point.md").write_text(
        "\n".join(
            [
                "# Selected validation operating point",
                "",
                f"- Confidence: `{selected['confidence']:.4f}`",
                f"- NMS IoU: `{args.iou}`",
                (
                    f"- Localized positive images: `{selected['localized_positive_images']} / "
                    f"{selected['positive_images']}`"
                ),
                f"- Box FN / FP: `{selected['box_fn']} / {selected['box_fp']}`",
                (
                    f"- Normal-image false alarms: `{selected['normal_false_alarm_images']} / "
                    f"{selected['normal_images']}`"
                ),
                "",
                "Selection was performed on validation only. Do not retune this value on test.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
