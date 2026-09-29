"""In-memory evidence images for the PDF report.

Nothing here touches disk: images are JPEG bytes held only for the duration of
an analysis, handed to the PDF generator, then cleared.
"""
from typing import Dict, List, Optional

import cv2
import numpy as np

from backend.models.schemas import PPE_LABEL

THUMB_HEIGHT = 240
EVENT_WIDTH = 960
MAX_EVENT_IMAGES = 60

RED = (40, 40, 220)
YELLOW = (0, 200, 250)


def _encode(img: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def _clip_box(b: List[float], w: int, h: int) -> List[int]:
    return [int(max(0, min(w, b[0]))), int(max(0, min(h, b[1]))), int(max(0, min(w, b[2]))), int(max(0, min(h, b[3])))]


class EvidenceStore:
    def __init__(self):
        self.thumbnails: Dict[int, bytes] = {}
        self._thumb_conf: Dict[int, float] = {}
        self.event_images: Dict[str, bytes] = {}

    def consider_thumbnail(self, track_id: int, conf: float, frame: np.ndarray, bbox: List[float]) -> bool:
        """Keep the highest-confidence crop of each worker. Returns True when replaced."""
        if conf <= self._thumb_conf.get(track_id, -1.0):
            return False
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        mx, my = 0.1 * (x2 - x1), 0.05 * (y2 - y1)
        cx1, cy1, cx2, cy2 = _clip_box([x1 - mx, y1 - my, x2 + mx, y2 + my], w, h)
        if cx2 - cx1 < 4 or cy2 - cy1 < 4:
            return False
        crop = frame[cy1:cy2, cx1:cx2]
        scale = THUMB_HEIGHT / crop.shape[0]
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), THUMB_HEIGHT), interpolation=cv2.INTER_AREA)
        self.thumbnails[track_id] = _encode(crop)
        self._thumb_conf[track_id] = conf
        return True

    def set_event_image(self, run_id: str, frame: np.ndarray, person_bbox: List[float],
                        item_bbox: Optional[List[float]], item: str) -> None:
        """Annotated, downscaled full frame for one violation run."""
        img = frame.copy()
        h, w = img.shape[:2]
        x1, y1, x2, y2 = _clip_box(person_bbox, w, h)
        thick = max(2, w // 400)
        cv2.rectangle(img, (x1, y1), (x2, y2), RED, thick)
        if item_bbox:
            ix1, iy1, ix2, iy2 = _clip_box(item_bbox, w, h)
            cv2.rectangle(img, (ix1, iy1), (ix2, iy2), YELLOW, thick)
        text = f"No {PPE_LABEL[item].lower()}"
        scale = max(0.5, w / 1400)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = max(th + base + 4, y1)
        cv2.rectangle(img, (x1, ty - th - base - 6), (x1 + tw + 8, ty), RED, -1)
        cv2.putText(img, text, (x1 + 4, ty - base - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
        if w > EVENT_WIDTH:
            img = cv2.resize(img, (EVENT_WIDTH, int(h * EVENT_WIDTH / w)), interpolation=cv2.INTER_AREA)
        self.event_images[run_id] = _encode(img)

    def drop(self, run_ids: List[str]) -> None:
        for rid in run_ids:
            self.event_images.pop(rid, None)

    def keep_only(self, run_ids: List[str]) -> None:
        keep = set(run_ids)
        for rid in list(self.event_images):
            if rid not in keep:
                del self.event_images[rid]

    def clear(self) -> None:
        self.thumbnails.clear()
        self._thumb_conf.clear()
        self.event_images.clear()
