from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_metrics(metrics: Any) -> dict[str, float | None]:
    """Extract stable scalar metrics from an Ultralytics validation result."""
    return {
        "box_precision": _number(getattr(metrics.box, "mp", None)),
        "box_recall": _number(getattr(metrics.box, "mr", None)),
        "box_map50": _number(getattr(metrics.box, "map50", None)),
        "box_map50_95": _number(getattr(metrics.box, "map", None)),
        "mask_precision": _number(getattr(metrics.seg, "mp", None)),
        "mask_recall": _number(getattr(metrics.seg, "mr", None)),
        "mask_map50": _number(getattr(metrics.seg, "map50", None)),
        "mask_map50_95": _number(getattr(metrics.seg, "map", None)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a stock YOLO segmentation checkpoint.")
    parser.add_argument("--model", required=True, help="Path to best.pt or an Ultralytics model name.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, help="Omit for standard mAP evaluation defaults.")
    parser.add_argument("--iou", type=float, default=0.7, help="Prediction NMS IoU.")
    parser.add_argument("--device")
    parser.add_argument("--project", type=Path, default=Path("reports/metrics"))
    parser.add_argument("--name", default="evaluation")
    parser.add_argument("--plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Missing data.yaml: {args.data}")

    import torch
    import ultralytics
    from ultralytics import YOLO

    validation_args: dict[str, Any] = {
        "data": str(args.data.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "iou": args.iou,
        "project": str(args.project.resolve()),
        "name": args.name,
        "plots": args.plots,
        "verbose": False,
    }
    if args.conf is not None:
        validation_args["conf"] = args.conf
    if args.device is not None:
        validation_args["device"] = args.device

    metrics = YOLO(args.model).val(**validation_args)
    summary = {
        "model": Path(args.model).name,
        "dataset": args.data.name,
        "split": args.split,
        "imgsz": args.imgsz,
        "confidence": args.conf,
        "nms_iou": args.iou,
        "metrics": collect_metrics(metrics),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
    }
    save_dir = Path(metrics.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "metrics_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
