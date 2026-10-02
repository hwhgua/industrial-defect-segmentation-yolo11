from __future__ import annotations

from pathlib import Path

import pytest

from ksdd2_seg.training import build_parser, build_training_plan, load_training_plan


def test_r4_config_is_stock_ultralytics_and_portable() -> None:
    root = Path(__file__).resolve().parents[1]
    config, detected_root = load_training_plan(root / "configs" / "r4_seed42.yaml")

    assert detected_root == root
    assert config["model"] == "yolo11n-seg.pt"
    assert config["train"]["mask_ratio"] == 4
    assert config["train"]["hsv_h"] == 0.0
    assert "device" not in config["train"]
    serialized = str(config)
    assert "P2ContextCalibratedSegment" not in serialized
    assert "P2SupervisedSegment" not in serialized


def test_explicit_training_config_is_independent_of_working_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    expected_root = Path(__file__).resolve().parents[1]
    config_path = expected_root / "configs" / "r4_seed42.yaml"

    _model, data, project, name, train_args, root = build_training_plan(
        build_parser().parse_args(["--config", str(config_path)])
    )

    assert root == expected_root
    assert data == expected_root / "datasets" / "kolektorsdd2_yolo_seg_formal" / "data.yaml"
    assert project == expected_root / "runs" / "train"
    assert name == "r4_seed42"
    assert train_args["seed"] == 42


def test_training_config_must_be_explicit_for_installed_package_portability() -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args([])

    assert error.value.code == 2
