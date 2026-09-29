"""Attach PPE / NO-PPE detections to tracked persons.

A PPE box is a candidate for a person when most of it (>= MIN_CONTAINMENT) lies
inside the person's box. Among candidates (overlapping workers), prefer the
person for whom the item sits closest to its expected body position: head items
near the top of the box, vests on the upper torso. The region prior only ranks
candidates, it never rejects one: in close-up shots the person box often extends
far beyond the frame and the face sits mid-box.
"""
from typing import Dict, List, Tuple

from backend.models.schemas import PPE_CLASSES, PPE_ITEMS, Detection, PPEObservation, TrackObservation
from backend.services.tracking import containment

MIN_CONTAINMENT = 0.5
# Expected vertical position of the item's centre, as a fraction of person height
EXPECTED_Y = {"hardhat": 0.06, "mask": 0.12, "vest": 0.35}
REGION_TOLERANCE = 0.45  # bonus fades to zero this far from the expected position
REGION_BONUS = 0.25
CENTER_PENALTY = 0.3


def _score(det: Detection, item: str, person_bbox: List[float]) -> float:
    c = containment(det.bbox, person_bbox)
    if c < MIN_CONTAINMENT:
        return -1.0
    px1, py1, px2, py2 = person_bbox
    pw, ph = max(1e-6, px2 - px1), max(1e-6, py2 - py1)
    cx = (det.bbox[0] + det.bbox[2]) / 2
    cy = (det.bbox[1] + det.bbox[3]) / 2
    ry = (cy - py1) / ph
    score = c + REGION_BONUS * max(0.0, 1.0 - abs(ry - EXPECTED_Y[item]) / REGION_TOLERANCE)
    score -= CENTER_PENALTY * abs(cx - (px1 + px2) / 2) / pw
    return score


def associate(
    tracks: List[Tuple[int, List[float], float]],
    detections: List[Detection],
) -> Tuple[List[TrackObservation], List[Detection]]:
    """Returns per-track PPE observations and the NO-* detections no person claimed."""
    best: Dict[int, Dict[str, Dict[str, Tuple[float, List[float]]]]] = {
        tid: {item: {} for item in PPE_ITEMS} for tid, _, _ in tracks
    }
    unattributed: List[Detection] = []
    for det in detections:
        mapping = PPE_CLASSES.get(det.class_name)
        if mapping is None:
            continue
        item, state = mapping
        scored = [(_score(det, item, bbox), tid) for tid, bbox, _ in tracks]
        scored = [s for s in scored if s[0] >= 0]
        if not scored:
            if state == "missing":
                unattributed.append(det)
            continue
        _, tid = max(scored)
        prev = best[tid][item].get(state)
        if prev is None or det.confidence > prev[0]:
            best[tid][item][state] = (det.confidence, det.bbox)

    observations = []
    for tid, bbox, conf in tracks:
        ppe = {}
        for item in PPE_ITEMS:
            present = best[tid][item].get("present")
            missing = best[tid][item].get("missing")
            if missing and (not present or missing[0] > present[0]):
                ppe[item] = PPEObservation("missing", missing[0], missing[1])
            elif present:
                ppe[item] = PPEObservation("present", present[0], present[1])
            else:
                ppe[item] = PPEObservation("unseen")
        observations.append(TrackObservation(track_id=tid, bbox=bbox, confidence=conf, ppe=ppe))
    return observations, unattributed
