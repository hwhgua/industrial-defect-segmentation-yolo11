from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run R4 inference on images or a directory.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.0975)
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--device")
    parser.add_argument("--project", type=Path, default=Path("runs/predict"))
    parser.add_argument("--name", default="r4")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from ultralytics import YOLO

    predict_args: dict[str, Any] = {
        "source": args.source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "project": str(args.project.resolve()),
        "name": args.name,
        "save": True,
    }
    if args.device is not None:
        predict_args["device"] = args.device
    YOLO(args.model).predict(**predict_args)


if __name__ == "__main__":
    main()
