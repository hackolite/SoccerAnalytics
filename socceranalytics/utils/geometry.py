"""
socceranalytics/utils/geometry.py
───────────────────────────────────
Geometry utilities — mirrors the top-level ``utils/bbox_utils.py`` but
scoped inside the package for clean imports.
"""
from __future__ import annotations

import math
from typing import Tuple


def get_center_of_bbox(bbox) -> Tuple[int, int]:
    """Return the (x, y) center of [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox) -> float:
    """Return the width of [x1, y1, x2, y2]."""
    return bbox[2] - bbox[0]


def get_foot_position(bbox) -> Tuple[int, int]:
    """Return the bottom-center (foot) position of [x1, y1, x2, y2]."""
    x1, _, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def measure_distance(p1, p2) -> float:
    """Euclidean distance between two 2-D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def measure_xy_distance(p1, p2) -> Tuple[float, float]:
    """Signed (dx, dy) displacement from p1 to p2."""
    return p2[0] - p1[0], p2[1] - p1[1]
