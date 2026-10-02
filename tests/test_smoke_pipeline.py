from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from ksdd2_seg.splitting import GENERATED_MARKER, prepare_dataset
from ksdd2_seg.verification import verify_dataset


def _write_pair(directory: Path, stem: str, defect: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 48, 3), 60, dtype=np.uint8)
    image[:, :, 1] += 20
    mask = np.zeros((32, 48), dtype=np.uint8)
    if defect:
        image[8:22, 12:34] = (160, 80, 40)
        mask[8:22, 12:34] = 255
    Image.fromarray(image).save(directory / f"{stem}.png")
    Image.fromarray(mask).save(directory / f"{stem}_GT.png")


def _make_source(root: Path) -> Path:
    source = root / "KolektorSDD2"
    for index in range(8):
        _write_pair(source / "train", f"1{index:03d}", defect=index < 4)
    _write_pair(source / "test", "2000", defect=True)
    _write_pair(source / "test", "2001", defect=False)
    return source


def test_synthetic_prepare_and_verify_pipeline(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    first_output = tmp_path / "prepared-one"
    second_output = tmp_path / "prepared-two"

    stats = prepare_dataset(source, first_output, val_ratio=0.25, seed=42)
    prepare_dataset(source, second_output, val_ratio=0.25, seed=42)

    assert stats["train"]["images"] == 6
    assert stats["val"]["images"] == 2
    assert stats["test"]["images"] == 2
    assert (first_output / GENERATED_MARKER).is_file()
    data_config = yaml.safe_load((first_output / "data.yaml").read_text(encoding="utf-8"))
    assert "path" not in data_config
    assert (first_output / data_config["train"]).is_dir()
    assert (first_output / data_config["val"]).is_dir()
    assert (first_output / data_config["test"]).is_dir()

    first_hash = hashlib.sha256((first_output / "split_manifest.csv").read_bytes()).hexdigest()
    second_hash = hashlib.sha256((second_output / "split_manifest.csv").read_bytes()).hexdigest()
    assert first_hash == second_hash

    report = verify_dataset(first_output)
    assert report.status == "PASS"
    assert report.errors == []


def test_verifier_rejects_out_of_range_coordinate(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "prepared"
    prepare_dataset(source, output, val_ratio=0.25, seed=42)
    defect_label = next(path for path in (output / "labels" / "train").glob("*.txt") if path.stat().st_size)
    defect_label.write_text("0 0.1 0.1 1.2 0.1 0.3 0.3\n", encoding="utf-8")

    report = verify_dataset(output)
    assert report.status == "FAIL"
    assert any("outside [0, 1]" in error for error in report.errors)


def test_verifier_rejects_ambiguous_dot_dataset_root(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "prepared"
    prepare_dataset(source, output, val_ratio=0.25, seed=42)
    data_yaml = output / "data.yaml"
    data_yaml.write_text("path: .\n" + data_yaml.read_text(encoding="utf-8"), encoding="utf-8")

    report = verify_dataset(output)
    assert report.status == "FAIL"
    assert any("must omit 'path'" in error for error in report.errors)


def test_prepare_rejects_image_mask_size_mismatch(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(source / "train" / "1000_GT.png")
    with pytest.raises(ValueError, match="size mismatch"):
        prepare_dataset(source, tmp_path / "prepared", val_ratio=0.25, seed=42)


def test_prepare_rejects_foreground_removed_by_area_threshold(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    tiny_mask = np.zeros((32, 48), dtype=np.uint8)
    tiny_mask[5, 5] = 255
    Image.fromarray(tiny_mask).save(source / "train" / "1000_GT.png")
    with pytest.raises(ValueError, match="produced no polygon"):
        prepare_dataset(source, tmp_path / "prepared", val_ratio=0.25, seed=42)


def test_failed_overwrite_preserves_previous_generated_dataset(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    output = tmp_path / "prepared"
    prepare_dataset(source, output, val_ratio=0.25, seed=42)
    sentinel = output / "keep-until-replacement-succeeds.txt"
    sentinel.write_text("old dataset is still intact\n", encoding="utf-8")

    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(source / "train" / "1000_GT.png")
    with pytest.raises(ValueError, match="size mismatch"):
        prepare_dataset(source, output, val_ratio=0.25, seed=42, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "old dataset is still intact\n"
    assert verify_dataset(output).status == "PASS"
