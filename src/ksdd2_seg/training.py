from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def find_project_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def load_training_plan(config_path: Path) -> tuple[dict[str, Any], Path]:
    config_path = config_path.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    required = {"model", "data", "project", "name", "train"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")
    if not isinstance(config["train"], dict):
        raise TypeError("The config 'train' entry must be a mapping.")
    return config, find_project_root(config_path)


def build_training_plan(args: argparse.Namespace) -> tuple[str, Path, Path, str, dict[str, Any], Path]:
    config, root = load_training_plan(args.config)
    model = args.model or str(config["model"])
    data = (args.data or (root / str(config["data"]))).expanduser().resolve()
    project = (args.project or (root / str(config["project"]))).expanduser().resolve()
    name = args.name or str(config["name"])
    train_args = dict(config["train"])
    for key in ("epochs", "seed", "batch", "workers", "imgsz"):
        value = getattr(args, key)
        if value is not None:
            train_args[key] = value
    if args.device is not None:
        train_args["device"] = args.device
    return model, data, project, name, train_args, root


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the stock-Ultralytics R4 YOLO11n-seg model.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Training YAML (the repository R4 config is configs/r4_seed42.yaml).",
    )
    parser.add_argument("--model", help="Ultralytics model name or local checkpoint.")
    parser.add_argument("--data", type=Path, help="Override data.yaml path.")
    parser.add_argument("--project", type=Path, help="Override generated run directory.")
    parser.add_argument("--name", help="Override run name.")
    parser.add_argument(
        "--device", help="Examples: 0, 0,1, cpu, mps. Omitted means Ultralytics auto-selection."
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the resolved plan without importing Ultralytics."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_name, data, project, name, train_args, root = build_training_plan(args)
    display_plan = {
        "model": model_name,
        "data": _display_path(data, root),
        "project": _display_path(project, root),
        "name": name,
        "train": train_args,
    }
    print(json.dumps(display_plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    if not data.is_file():
        raise FileNotFoundError(f"Missing dataset YAML: {data}")

    # Delayed import keeps config validation and --help usable without GPU packages.
    from ultralytics import YOLO

    model = YOLO(model_name)
    model.train(data=str(data), project=str(project), name=name, **train_args)


if __name__ == "__main__":
    main()
