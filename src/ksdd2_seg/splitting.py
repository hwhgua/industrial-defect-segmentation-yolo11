from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .conversion import mask_to_segments, yolo_line

GENERATED_MARKER = ".ksdd2-seg-generated"


@dataclass(frozen=True)
class SourceSample:
    image_path: Path
    mask_path: Path
    source_split: str
    is_defect: bool


def collect_samples(source_split_dir: Path, source_split: str) -> list[SourceSample]:
    """Collect image/mask pairs and classify them as defect or normal."""
    if not source_split_dir.is_dir():
        raise FileNotFoundError(f"Missing source split directory: {source_split_dir}")

    samples: list[SourceSample] = []
    image_paths = sorted(path for path in source_split_dir.glob("*.png") if not path.name.endswith("_GT.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {source_split_dir}")

    for image_path in image_paths:
        mask_path = image_path.with_name(f"{image_path.stem}_GT.png")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}")
        with Image.open(image_path) as image:
            width, height = image.size
        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image.convert("L"))
        if mask.shape != (height, width):
            raise ValueError(
                f"Image/mask size mismatch for {image_path.name}: "
                f"image={(width, height)}, mask={(mask.shape[1], mask.shape[0])}"
            )
        samples.append(
            SourceSample(
                image_path=image_path,
                mask_path=mask_path,
                source_split=source_split,
                is_defect=bool((mask > 0).any()),
            )
        )
    return samples


def stratified_train_val_split(
    samples: list[SourceSample], val_ratio: float, seed: int
) -> tuple[list[SourceSample], list[SourceSample]]:
    """Split original training data while preserving defect/normal proportions."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if len(samples) < 2:
        raise ValueError("At least two training samples are required.")

    rng = random.Random(seed)
    defect_samples = [sample for sample in samples if sample.is_defect]
    normal_samples = [sample for sample in samples if not sample.is_defect]
    rng.shuffle(defect_samples)
    rng.shuffle(normal_samples)

    def split_group(group: list[SourceSample]) -> tuple[list[SourceSample], list[SourceSample]]:
        if len(group) <= 1:
            return group, []
        val_count = min(len(group) - 1, max(1, round(len(group) * val_ratio)))
        return group[val_count:], group[:val_count]

    train_defect, val_defect = split_group(defect_samples)
    train_normal, val_normal = split_group(normal_samples)
    train = sorted(train_defect + train_normal, key=lambda sample: sample.image_path.name)
    val = sorted(val_defect + val_normal, key=lambda sample: sample.image_path.name)
    if not train or not val:
        raise ValueError("The requested split produced an empty train or validation set.")
    return train, val


def convert_samples(samples: list[SourceSample], output_dir: Path, split_name: str) -> dict[str, int]:
    """Copy images and convert masks into YOLO polygon labels."""
    image_out = output_dir / "images" / split_name
    label_out = output_dir / "labels" / split_name
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "defect_images": 0, "normal_images": 0, "segments": 0}
    for sample in samples:
        with Image.open(sample.image_path) as image:
            width, height = image.size
        with Image.open(sample.mask_path) as mask_image:
            mask = np.asarray(mask_image.convert("L"))
        if mask.shape != (height, width):
            raise ValueError(
                f"Image/mask size mismatch for {sample.image_path.name}: "
                f"image={(width, height)}, mask={(mask.shape[1], mask.shape[0])}"
            )
        polygons = mask_to_segments(mask)
        mask_has_foreground = bool((mask > 0).any())
        if mask_has_foreground and not polygons:
            raise ValueError(
                f"Foreground mask for {sample.image_path.name} produced no polygon. "
                "Review the mask or conversion area threshold."
            )
        if mask_has_foreground != sample.is_defect:
            raise ValueError(f"Mask content changed after sample collection: {sample.image_path.name}")

        shutil.copy2(sample.image_path, image_out / sample.image_path.name)
        label_path = label_out / f"{sample.image_path.stem}.txt"
        if polygons:
            lines = [yolo_line(polygon, width, height) for polygon in polygons]
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            stats["defect_images"] += 1
            stats["segments"] += len(polygons)
        else:
            label_path.write_text("", encoding="utf-8")
            stats["normal_images"] += 1
        stats["images"] += 1
    return stats


def write_data_yaml(output_dir: Path) -> None:
    (output_dir / "data.yaml").write_text(
        "\n".join(
            [
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: defect",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_manifest(output_dir: Path, splits: dict[str, list[SourceSample]]) -> str:
    manifest_path = output_dir / "split_manifest.csv"
    fieldnames = ["split", "image_name", "label_name", "source_split", "is_defect"]
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\r\n")
        writer.writeheader()
        for split_name in ("train", "val", "test"):
            for sample in splits[split_name]:
                writer.writerow(
                    {
                        "split": split_name,
                        "image_name": sample.image_path.name,
                        "label_name": f"{sample.image_path.stem}.txt",
                        "source_split": sample.source_split,
                        "is_defect": int(sample.is_defect),
                    }
                )
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest().upper()


def write_summary(
    output_dir: Path,
    val_ratio: float,
    seed: int,
    stats: dict[str, dict[str, int]],
    manifest_sha256: str,
) -> None:
    lines = [
        "# KolektorSDD2 formal YOLO split",
        "",
        "Source layout: `train/*.png`, `train/*_GT.png`, `test/*.png`, `test/*_GT.png`.",
        f"Validation ratio from original train: `{val_ratio}`",
        f"Random seed: `{seed}`",
        f"Manifest SHA-256: `{manifest_sha256}`",
        "",
        "| split | images | defect images | normal images | segments | source |",
        "|---|---:|---:|---:|---:|---|",
    ]
    sources = {
        "train": "original train subset",
        "val": "original train subset",
        "test": "original test",
    }
    for split_name in ("train", "val", "test"):
        row = stats[split_name]
        lines.append(
            f"| {split_name} | {row['images']} | {row['defect_images']} | "
            f"{row['normal_images']} | {row['segments']} | {sources[split_name]} |"
        )
    lines.extend(
        [
            "",
            "Empty label files represent normal images with no defect mask.",
            "The source dataset is not redistributed by this project.",
        ]
    )
    (output_dir / "formal_split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_source(source: Path) -> Path:
    source = source.expanduser().resolve()
    for split_name in ("train", "test"):
        if not (source / split_name).is_dir():
            raise FileNotFoundError(f"Source must contain a {split_name!r} directory: {source}")
    return source


def _prepare_output(output: Path, source: Path, overwrite: bool) -> Path:
    output = output.expanduser().resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("Output must not replace the source dataset.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
        if not (output / GENERATED_MARKER).is_file():
            raise ValueError(
                f"Refusing to delete an unmarked directory: {output}. "
                "Only outputs created by this tool can be overwritten."
            )
    return output


def prepare_dataset(
    source: Path,
    output: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    """Build the formal train/validation/test dataset atomically."""
    source = _validate_source(source)
    output = _prepare_output(output, source, overwrite)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        original_train = collect_samples(source / "train", "train")
        original_test = collect_samples(source / "test", "test")
        train_samples, val_samples = stratified_train_val_split(original_train, val_ratio, seed)
        splits = {"train": train_samples, "val": val_samples, "test": original_test}
        stats = {
            split_name: convert_samples(samples, staging, split_name)
            for split_name, samples in splits.items()
        }
        write_data_yaml(staging)
        manifest_sha256 = write_manifest(staging, splits)
        write_summary(staging, val_ratio, seed, stats, manifest_sha256)
        (staging / GENERATED_MARKER).write_text(
            "Generated by ksdd2-prepare. Safe for this tool to replace with --overwrite.\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    backup: Path | None = None
    try:
        if output.exists():
            backup = output.with_name(f".{output.name}-backup-{uuid.uuid4().hex}")
            output.replace(backup)
        staging.replace(output)
    except Exception:
        if backup is not None and backup.exists() and not output.exists():
            backup.replace(output)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    else:
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic YOLO train/val/test split from KolektorSDD2."
    )
    parser.add_argument("--source", type=Path, required=True, help="Extracted KolektorSDD2 directory.")
    parser.add_argument("--output", type=Path, required=True, help="Destination dataset directory.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = prepare_dataset(args.source, args.output, args.val_ratio, args.seed, args.overwrite)
    for split_name, row in stats.items():
        print(f"{split_name}: {row}")


if __name__ == "__main__":
    main()
