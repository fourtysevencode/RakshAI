"""Approximate (2D, image-space) proximity between workers and machinery/vehicles."""
import math
from typing import List

from backend.models.schemas import HAZARD_CLASSES, Detection

# A worker is "near" a hazard when the gap between boxes is below this fraction
# of the worker's box height (a rough, perspective-agnostic scale reference).
NEAR_FACTOR = 0.5


def box_gap(a: List[float], b: List[float]) -> float:
    """Euclidean gap between two boxes; 0 when they overlap."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def nearby_hazards(person_bbox: List[float], detections: List[Detection]) -> List[str]:
    height = max(1.0, person_bbox[3] - person_bbox[1])
    near = set()
    for d in detections:
        kind = HAZARD_CLASSES.get(d.class_name)
        if kind and box_gap(person_bbox, d.bbox) < NEAR_FACTOR * height:
            near.add(kind)
    return sorted(near)
