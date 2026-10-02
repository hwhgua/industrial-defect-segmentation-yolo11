from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from PIL import Image


@dataclass
class SplitStats:
    split: str
    images: int = 0
    labels: int = 0
    defect_images: int = 0
    normal_images: int = 0
    segments: int = 0
    invalid_labels: int = 0
    missing_labels: int = 0
    missing_images: int = 0


@dataclass
class VerificationReport:
    status: str
    stats: list[SplitStats]
    errors: list[str]


def validate_label_file(label_path: Path) -> tuple[int, list[str]]:
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return 0, []

    errors: list[str] = []
    segment_count = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        prefix = f"{label_path.name}:{line_number}"
        if len(parts) < 7:
            errors.append(f"{prefix} has fewer than three polygon points")
            continue
        if (len(parts) - 1) % 2:
            errors.append(f"{prefix} has an uneven coordinate count")
            continue
        if parts[0] != "0":
            errors.append(f"{prefix} has unexpected class id {parts[0]}")
        try:
            coordinates = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{prefix} has non-numeric coordinates")
            continue
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            errors.append(f"{prefix} has coordinates outside [0, 1]")
        segment_count += 1
    return segment_count, errors


def verify_split(dataset_dir: Path, split_name: str) -> tuple[SplitStats, list[str], set[str]]:
    image_dir = dataset_dir / "images" / split_name
    label_dir = dataset_dir / "labels" / split_name
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing images/labels directories for split {split_name!r}.")

    image_paths = sorted(image_dir.glob("*.png"))
    label_paths = sorted(label_dir.glob("*.txt"))
    image_stems = {path.stem for path in image_paths}
    label_stems = {path.stem for path in label_paths}
    stats = SplitStats(split=split_name, images=len(image_paths), labels=len(label_paths))
    errors: list[str] = []

    stats.missing_labels = len(image_stems - label_stems)
    stats.missing_images = len(label_stems - image_stems)
    for stem in sorted(image_stems - label_stems):
        errors.append(f"{split_name}: missing label for {stem}.png")
    for stem in sorted(label_stems - image_stems):
        errors.append(f"{split_name}: missing image for {stem}.txt")

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as error:  # noqa: BLE001
            errors.append(f"{split_name}: unreadable image {image_path.name}: {error}")

    for label_path in label_paths:
        segment_count, label_errors = validate_label_file(label_path)
        stats.segments += segment_count
        stats.invalid_labels += len(label_errors)
        errors.extend(f"{split_name}: {error}" for error in label_errors)
        if segment_count:
            stats.defect_images += 1
        else:
            stats.normal_images += 1
    return stats, errors, image_stems


def verify_dataset(dataset_dir: Path) -> VerificationReport:
    dataset_dir = dataset_dir.expanduser().resolve()
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Missing data.yaml: {data_yaml}")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    expected = {"train": "images/train", "val": "images/val", "test": "images/test"}
    errors = [
        f"data.yaml {key!r}: expected {value!r}, got {config.get(key)!r}"
        for key, value in expected.items()
        if config.get(key) != value
    ]
    if config.get("path"):
        errors.append(
            "data.yaml must omit 'path' so Ultralytics resolves image paths relative to the YAML file; "
            f"got {config['path']!r}"
        )

    stats_rows: list[SplitStats] = []
    stems_by_split: dict[str, set[str]] = {}
    for split_name in ("train", "val", "test"):
        stats, split_errors, stems = verify_split(dataset_dir, split_name)
        stats_rows.append(stats)
        errors.extend(split_errors)
        stems_by_split[split_name] = stems

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = stems_by_split[left] & stems_by_split[right]
        if shared:
            preview = ", ".join(sorted(shared)[:5])
            errors.append(f"{left}/{right} share {len(shared)} image stems: {preview}")
    return VerificationReport(status="PASS" if not errors else "FAIL", stats=stats_rows, errors=errors)


def write_report(report: VerificationReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "split_verification.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SplitStats.__dataclass_fields__.keys())
        writer.writeheader()
        writer.writerows(asdict(row) for row in report.stats)
    (output_dir / "verification_errors.txt").write_text(
        "\n".join(report.errors) + ("\n" if report.errors else ""), encoding="utf-8"
    )

    lines = [
        "# YOLO segmentation dataset verification",
        "",
        f"Status: **{report.status}**",
        "",
        (
            "| split | images | labels | defect | normal | segments | missing labels | "
            "missing images | invalid labels |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.stats:
        lines.append(
            f"| {row.split} | {row.images} | {row.labels} | {row.defect_images} | "
            f"{row.normal_images} | {row.segments} | {row.missing_labels} | "
            f"{row.missing_images} | {row.invalid_labels} |"
        )
    lines.extend(
        [
            "",
            f"Total errors: `{len(report.errors)}`",
            "",
            (
                "Checks: image/label pairing, image readability, class id, polygon shape, "
                "normalized coordinates, and cross-split stem overlap."
            ),
        ]
    )
    (output_dir / "verification_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a prepared YOLO segmentation dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/metrics/dataset_verification"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = verify_dataset(args.dataset)
    write_report(report, args.output)
    print(f"Dataset verification: {report.status}; errors={len(report.errors)}")
    if report.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
