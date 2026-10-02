from __future__ import annotations

import numpy as np
import pytest

from ksdd2_seg.conversion import mask_to_segments, polygon_area, yolo_line


def test_blank_mask_has_no_segments() -> None:
    assert mask_to_segments(np.zeros((24, 32), dtype=np.uint8)) == []


def test_rectangle_becomes_normalized_polygon() -> None:
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[5:18, 8:26] = 255
    segments = mask_to_segments(mask, simplify_every=2, min_area=4)

    assert len(segments) == 1
    assert polygon_area(segments[0]) >= 4
    values = yolo_line(segments[0], width=32, height=24).split()
    assert values[0] == "0"
    assert len(values) >= 7
    assert all(0.0 <= float(value) <= 1.0 for value in values[1:])


def test_multiple_components_remain_separate() -> None:
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[4:14, 5:15] = 255
    mask[24:36, 25:37] = 255
    assert len(mask_to_segments(mask, simplify_every=2, min_area=4)) == 2


def test_material_hole_is_rejected_instead_of_becoming_an_object() -> None:
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5:35, 5:35] = 255
    mask[15:25, 15:25] = 0
    with pytest.raises(ValueError, match="enclosed hole"):
        mask_to_segments(mask, simplify_every=2, min_area=4)


def test_border_touching_defect_is_supported() -> None:
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[0:12, 0:18] = 255
    segments = mask_to_segments(mask, simplify_every=2, min_area=4)
    assert len(segments) == 1
    values = [float(value) for value in yolo_line(segments[0], 30, 30).split()[1:]]
    assert all(0.0 <= value <= 1.0 for value in values)
