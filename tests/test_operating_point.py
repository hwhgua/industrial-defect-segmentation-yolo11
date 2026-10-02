from __future__ import annotations

from ksdd2_seg.select_operating_point import confidence_grid, select_best


def test_confidence_grid_contains_reference_threshold_exactly() -> None:
    values = confidence_grid(0.02, 0.10, 0.0025)
    assert 0.0975 in values
    assert values[0] == 0.02
    assert values[-1] == 0.10


def test_selection_respects_alarm_budget_before_recall() -> None:
    rows = [
        {
            "confidence": 0.05,
            "localized_positive_images": 40,
            "box_fn": 10,
            "box_f1": 0.7,
            "box_precision": 0.6,
            "normal_false_alarm_images": 8,
        },
        {
            "confidence": 0.0975,
            "localized_positive_images": 35,
            "box_fn": 15,
            "box_f1": 0.65,
            "box_precision": 0.61,
            "normal_false_alarm_images": 4,
        },
    ]
    assert select_best(rows, max_normal_alarms=5)["confidence"] == 0.0975
