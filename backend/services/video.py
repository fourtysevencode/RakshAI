"""Video reading and time-based frame sampling.

Sampling is driven by each frame's real timestamp (CAP_PROP_POS_MSEC), not its
index: browser-recorded .webm files are variable frame rate and OpenCV reports
them as fps=1000 with a "frame count" that is really a millisecond count.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np

MAX_W, MAX_H = 1280, 720


@dataclass
class VideoInfo:
    fps: float  # container-reported; unreliable for VFR, informational only
    duration: float  # seconds
    width: int
    height: int
    proc_width: int  # size of the frames handed to the model
    proc_height: int


def _proc_size(w: int, h: int) -> Tuple[int, int]:
    if w <= MAX_W and h <= MAX_H:
        return w, h
    scale = min(MAX_W / w, MAX_H / h)
    return max(1, round(w * scale)), max(1, round(h * scale))


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        # count/fps is right even for VFR webm (both are in ms units there)
        duration = count / fps if fps > 0 else 0.0
        pw, ph = _proc_size(w, h)
        return VideoInfo(fps=fps, duration=duration, width=w, height=h, proc_width=pw, proc_height=ph)
    finally:
        cap.release()


def iter_sampled_frames(path: Path, analysis_fps: float) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Yield (frame_idx, t_seconds, frame) roughly every 1/analysis_fps seconds.

    analysis_fps <= 0 yields every frame. Skipped frames are only grabbed, not
    retrieved/converted. Yielded frames are resized to fit 1280x720.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {path}")
    step = 1.0 / analysis_fps if analysis_fps and analysis_fps > 0 else 0.0
    next_t = 0.0
    frame_idx = -1
    try:
        while cap.grab():
            frame_idx += 1
            t = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            if step and t + 1e-6 < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            h, w = frame.shape[:2]
            pw, ph = _proc_size(w, h)
            if (pw, ph) != (w, h):
                frame = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
            if step:
                # Don't try to "catch up" after a long gap between frames
                next_t = max(next_t + step, t + step / 2)
            yield frame_idx, t, frame
    finally:
        cap.release()
