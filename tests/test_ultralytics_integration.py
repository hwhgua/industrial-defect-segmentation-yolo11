from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ksdd2_seg.splitting import prepare_dataset

pytestmark = pytest.mark.integration

# Keep Ultralytics test state inside the repository's ignored run directory.
_ultralytics_config = Path(__file__).resolve().parents[1] / "runs" / "ultralytics-config"
_ultralytics_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ultralytics_config))
pytest.importorskip("ultralytics")


def _write_pair(directory: Path, stem: str, defect: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    image = np.full((24, 32, 3), 80, dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=np.uint8)
    if defect:
        image[6:18, 9:24] = (150, 70, 30)
        mask[6:18, 9:24] = 255
    Image.fromarray(image).save(directory / f"{stem}.png")
    Image.fromarray(mask).save(directory / f"{stem}_GT.png")


def _make_prepared_dataset(tmp_path: Path) -> Path:
    source = tmp_path / "raw"
    for index in range(8):
        _write_pair(source / "train", f"1{index:03d}", defect=index < 4)
    _write_pair(source / "test", "2000", defect=True)
    _write_pair(source / "test", "2001", defect=False)
    prepared = tmp_path / "prepared"
    prepare_dataset(source, prepared, val_ratio=0.25, seed=42)
    return prepared


def test_generated_yaml_resolves_from_its_own_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ultralytics.data.utils as data_utils

    prepared = _make_prepared_dataset(tmp_path)

    # Dataset path resolution should not need a font download.
    monkeypatch.setattr(data_utils, "check_font", lambda *_args, **_kwargs: None)
    resolved = data_utils.check_det_dataset(str(prepared / "data.yaml"), autodownload=False)

    assert Path(resolved["path"]).resolve() == prepared.resolve()
    assert Path(resolved["train"]).resolve() == (prepared / "images" / "train").resolve()
    assert Path(resolved["val"]).resolve() == (prepared / "images" / "val").resolve()
    assert Path(resolved["test"]).resolve() == (prepared / "images" / "test").resolve()


def test_one_epoch_stock_ultralytics_training_smoke(tmp_path: Path) -> None:
    prepared = _make_prepared_dataset(tmp_path)
    project_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "runs"
    command = [
        sys.executable,
        "-m",
        "ksdd2_seg.training",
        "--config",
        str(project_root / "configs" / "r4_seed42.yaml"),
        "--model",
        "yolo11n-seg.yaml",
        "--data",
        str(prepared / "data.yaml"),
        "--project",
        str(run_root),
        "--name",
        "cpu_one_epoch",
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--batch",
        "2",
        "--workers",
        "0",
        "--imgsz",
        "64",
    ]
    subprocess.run(command, cwd=project_root, check=True)

    run_dir = run_root / "cpu_one_epoch"
    assert (run_dir / "results.csv").is_file()
    assert (run_dir / "weights" / "best.pt").is_file()
    assert (run_dir / "weights" / "last.pt").is_file()
