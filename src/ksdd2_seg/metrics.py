from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Instance:
    box: Box
    mask: np.ndarray
    confidence: float = 1.0


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def box_iou(first: Box, second: Box) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0


def dice(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    total = first.sum() + second.sum()
    return float(2 * intersection / total) if total else 0.0


def metric_matrix(
    ground_truth: list[Instance],
    predictions: list[Instance],
    metric: Callable[[Any, Any], float],
    attribute: str,
) -> np.ndarray:
    matrix = np.zeros((len(ground_truth), len(predictions)), dtype=np.float32)
    for gt_index, gt_instance in enumerate(ground_truth):
        for pred_index, pred_instance in enumerate(predictions):
            matrix[gt_index, pred_index] = metric(
                getattr(gt_instance, attribute), getattr(pred_instance, attribute)
            )
    return matrix


def greedy_match(matrix: np.ndarray, threshold: float) -> dict[int, int]:
    """Greedily create one-to-one matches in descending score order."""
    candidates = [
        (float(matrix[gt_index, pred_index]), gt_index, pred_index)
        for gt_index in range(matrix.shape[0])
        for pred_index in range(matrix.shape[1])
        if float(matrix[gt_index, pred_index]) >= threshold
    ]
    matches: dict[int, int] = {}
    used_predictions: set[int] = set()
    for _score, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index not in matches and pred_index not in used_predictions:
            matches[gt_index] = pred_index
            used_predictions.add(pred_index)
    return matches


def _polygon_mask(points: list[tuple[float, float]], width: int, height: int) -> np.ndarray:
    canvas = Image.new("1", (width, height), 0)
    if len(points) >= 3:
        ImageDraw.Draw(canvas).polygon(points, fill=1)
    return np.asarray(canvas, dtype=bool)


def read_label_instances(label_path: Path, width: int, height: int) -> list[Instance]:
    if not label_path.is_file() or not label_path.read_text(encoding="utf-8").strip():
        return []
    instances: list[Instance] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        coordinates = [float(value) for value in parts[1:]]
        points = [
            (coordinates[index] * width, coordinates[index + 1] * height)
            for index in range(0, len(coordinates), 2)
        ]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        instances.append(
            Instance(
                box=(min(xs), min(ys), max(xs), max(ys)),
                mask=_polygon_mask(points, width, height),
            )
        )
    return instances


def read_prediction_instances(result: Any, width: int, height: int) -> list[Instance]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    mask_polygons = result.masks.xy if result.masks is not None else []
    instances: list[Instance] = []
    for index, box_values in enumerate(boxes):
        box = tuple(float(value) for value in box_values)
        if index < len(mask_polygons) and len(mask_polygons[index]) >= 3:
            points = [(float(x), float(y)) for x, y in mask_polygons[index]]
            mask = _polygon_mask(points, width, height)
        else:
            x1, y1, x2, y2 = box
            mask = _polygon_mask([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], width, height)
        instances.append(Instance(box=box, mask=mask, confidence=float(confidences[index])))
    return instances
