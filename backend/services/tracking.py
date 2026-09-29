"""Person tracking with Ultralytics' ByteTrack, fed by our own detections.

Feeding BYTETracker directly (instead of model.track()) keeps tracking
independent of the inference backend, so it works with both the Ultralytics
and the plain onnxruntime path in backend/utils/predict.py.
"""
from argparse import Namespace
from typing import List, Tuple

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from backend.models.schemas import Detection


class _Dets:
    """Minimal Results-like adapter: BYTETracker needs conf, xywh, cls and boolean indexing."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy.reshape(-1, 4).astype(np.float32)
        self.conf = conf.astype(np.float32)
        self.cls = cls.astype(np.float32)

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_Dets":
        return _Dets(self.xyxy[idx], self.conf[idx], self.cls[idx])


def containment(inner: List[float], outer: List[float]) -> float:
    """Fraction of `inner`'s area that lies inside `outer`."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return ix * iy / area


def suppress_duplicate_persons(persons: List[Detection], thresh: float = 0.85) -> List[Detection]:
    """Drop a person box when it mostly contains, or is mostly contained by, a
    higher-confidence one. The model often emits a second, looser box around the
    same worker that standard IoU-NMS keeps, which would otherwise spawn a
    phantom track."""
    kept: List[Detection] = []
    for p in sorted(persons, key=lambda d: -d.confidence):
        if all(containment(p.bbox, k.bbox) < thresh and containment(k.bbox, p.bbox) < thresh for k in kept):
            kept.append(p)
    return kept


class PersonTracker:
    def __init__(self, sample_fps: float, buffer_seconds: float = 2.0):
        args = Namespace(
            tracker_type="bytetrack",
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=max(3, int(round(buffer_seconds * sample_fps))),
            match_thresh=0.8,
            fuse_score=True,
        )
        self._tracker = BYTETracker(args)
        self._tracker.reset()  # track IDs are a class-level counter; start each analysis at 1

    def update(self, persons: List[Detection]) -> List[Tuple[int, List[float], float]]:
        """Returns [(track_id, bbox, confidence)] for persons tracked in this frame."""
        if persons:
            xyxy = np.array([p.bbox for p in persons], dtype=np.float32)
            conf = np.array([p.confidence for p in persons], dtype=np.float32)
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
        out = self._tracker.update(_Dets(xyxy, conf, np.zeros_like(conf)))
        tracked = []
        for row in out:
            x1, y1, x2, y2, tid, score = row[:6]
            tracked.append((int(tid), [float(x1), float(y1), float(x2), float(y2)], float(score)))
        return tracked
