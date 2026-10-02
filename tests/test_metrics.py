from __future__ import annotations

import numpy as np

from ksdd2_seg.analyze import evaluate_instances, summarize
from ksdd2_seg.metrics import Instance, box_iou, greedy_match


def _instance(box: tuple[float, float, float, float]) -> Instance:
    mask = np.zeros((20, 20), dtype=bool)
    x1, y1, x2, y2 = (int(value) for value in box)
    mask[y1:y2, x1:x2] = True
    return Instance(box=box, mask=mask, confidence=0.8)


def test_box_iou_and_one_to_one_matching() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    matrix = np.array([[0.9, 0.8], [0.7, 0.1]], dtype=np.float32)
    assert greedy_match(matrix, threshold=0.5) == {0: 0}


def test_operating_point_summary_counts_normal_alarm() -> None:
    matched_row, _, _ = evaluate_instances([_instance((2, 2, 10, 10))], [_instance((2, 2, 10, 10))], 0.5, 0.3)
    normal_alarm_row, _, _ = evaluate_instances([], [_instance((4, 4, 8, 8))], 0.5, 0.3)
    summary = summarize([matched_row, normal_alarm_row])

    assert summary["box_precision"] == 0.5
    assert summary["box_recall"] == 1.0
    assert summary["localized_positive_images"] == 1
    assert summary["normal_false_alarm_images"] == 1
    assert summary["normal_image_false_alarm_rate"] == 1.0
