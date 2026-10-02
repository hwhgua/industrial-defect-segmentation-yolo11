from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

Point = tuple[float, float]
Polygon = list[Point]


def _binary_mask_has_holes(binary: np.ndarray) -> bool:
    """Return True when background pixels are enclosed by foreground.

    The check uses a padded, four-connected flood fill from the exterior. It
    is called only for masks that produced multiple material contours.
    """
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    background = ~padded
    visited = np.zeros_like(background, dtype=bool)
    queue: deque[tuple[int, int]] = deque([(0, 0)])
    visited[0, 0] = True
    max_y, max_x = background.shape
    while queue:
        y, x = queue.popleft()
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                0 <= next_y < max_y
                and 0 <= next_x < max_x
                and background[next_y, next_x]
                and not visited[next_y, next_x]
            ):
                visited[next_y, next_x] = True
                queue.append((next_y, next_x))
    return bool(np.any(background & ~visited))


def polygon_area(points: Polygon) -> float:
    """Return the unsigned shoelace area of a polygon."""
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def mask_to_segments(mask: np.ndarray, simplify_every: int = 4, min_area: int = 8) -> list[Polygon]:
    """Convert a binary mask to approximate polygons using marching squares.

    Coordinates are returned in source-image pixel units. The implementation
    intentionally depends only on NumPy so dataset preparation does not need
    OpenCV or a custom Ultralytics fork.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {mask.shape}.")
    if simplify_every < 1:
        raise ValueError("simplify_every must be at least 1.")
    if min_area < 0:
        raise ValueError("min_area must be non-negative.")

    binary = mask > 0
    if not binary.any():
        return []

    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    height, width = binary.shape
    segments: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def edge_points(x: int, y: int) -> dict[str, tuple[int, int]]:
        return {
            "T": (2 * x + 1, 2 * y),
            "R": (2 * x + 2, 2 * y + 1),
            "B": (2 * x + 1, 2 * y + 2),
            "L": (2 * x, 2 * y + 1),
        }

    case_segments = {
        1: [("L", "B")],
        2: [("B", "R")],
        3: [("L", "R")],
        4: [("T", "R")],
        5: [("T", "L"), ("B", "R")],
        6: [("T", "B")],
        7: [("T", "L")],
        8: [("L", "T")],
        9: [("T", "B")],
        10: [("T", "R"), ("L", "B")],
        11: [("T", "R")],
        12: [("L", "R")],
        13: [("R", "B")],
        14: [("B", "L")],
    }

    for y in range(padded.shape[0] - 1):
        for x in range(padded.shape[1] - 1):
            top_left = int(padded[y, x])
            top_right = int(padded[y, x + 1])
            bottom_right = int(padded[y + 1, x + 1])
            bottom_left = int(padded[y + 1, x])
            case = (top_left << 3) | (top_right << 2) | (bottom_right << 1) | bottom_left
            if case in {0, 15}:
                continue
            points = edge_points(x, y)
            for first, second in case_segments[case]:
                segments.append((points[first], points[second]))

    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    unused_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for first, second in segments:
        adjacency[first].append(second)
        adjacency[second].append(first)
        unused_edges.add(tuple(sorted((first, second))))

    contours: list[Polygon] = []
    while unused_edges:
        start, current = unused_edges.pop()
        previous = start
        contour = [start, current]

        while current != start:
            next_point = None
            for candidate in adjacency[current]:
                candidate_edge = tuple(sorted((current, candidate)))
                if candidate != previous and candidate_edge in unused_edges:
                    next_point = candidate
                    break
            if next_point is None:
                break
            unused_edges.remove(tuple(sorted((current, next_point))))
            previous, current = current, next_point
            contour.append(current)

        if len(contour) < 4:
            continue

        polygon: Polygon = []
        for point_x, point_y in contour:
            x = max(0.0, min(float(width), point_x / 2.0 - 1.0))
            y = max(0.0, min(float(height), point_y / 2.0 - 1.0))
            if not polygon or polygon[-1] != (x, y):
                polygon.append((x, y))

        if len(polygon) > simplify_every * 3:
            polygon = polygon[::simplify_every]
        if len(polygon) >= 3 and polygon_area(polygon) >= min_area:
            contours.append(polygon)

    if len(contours) > 1 and _binary_mask_has_holes(binary):
        raise ValueError(
            "Mask contains an enclosed hole that cannot be represented faithfully "
            "by this YOLO polygon converter."
        )

    return contours


def yolo_line(points: Polygon, width: int, height: int, class_id: int = 0) -> str:
    """Serialize one polygon as a normalized YOLO segmentation label."""
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")
    values = [str(class_id)]
    for x, y in points:
        values.extend((f"{x / width:.6f}", f"{y / height:.6f}"))
    return " ".join(values)
