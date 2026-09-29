"""Plain dataclasses shared by the analysis services.

Boxes are [x1, y1, x2, y2] in pixels of the *processed* frame (downscaled to at
most 1280x720). The API converts them to 0-1 normalized coordinates so the UI
can draw on the original video without knowing the processing size.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# PPE items the model can judge, and the classes that indicate each state.
PPE_ITEMS = ("hardhat", "vest", "mask")
PPE_CLASSES = {
    "Hardhat": ("hardhat", "present"),
    "NO-Hardhat": ("hardhat", "missing"),
    "Safety Vest": ("vest", "present"),
    "NO-Safety Vest": ("vest", "missing"),
    "Mask": ("mask", "present"),
    "NO-Mask": ("mask", "missing"),
}
VIOLATION_TYPE = {"hardhat": "NO_HARDHAT", "vest": "NO_VEST", "mask": "NO_MASK"}
PPE_LABEL = {"hardhat": "Hardhat", "vest": "Safety vest", "mask": "Mask"}
HAZARD_CLASSES = {"machinery": "machinery", "vehicle": "vehicle"}
PERSON_CLASS = "Person"
CONE_CLASS = "Safety Cone"


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]

    @classmethod
    def from_dict(cls, d: dict) -> "Detection":
        return cls(int(d["class_id"]), d["class_name"], float(d["confidence"]), [float(v) for v in d["bbox"]])


@dataclass
class PPEObservation:
    state: str  # present | missing | unseen
    confidence: float = 0.0
    bbox: Optional[List[float]] = None


@dataclass
class TrackObservation:
    """One tracked person in one sampled frame."""
    track_id: int
    bbox: List[float]
    confidence: float
    ppe: Dict[str, PPEObservation]
    near: List[str] = field(default_factory=list)  # hazard kinds nearby, e.g. ["machinery"]


@dataclass
class FrameSample:
    frame_idx: int
    t: float  # seconds from video start (real timestamp, VFR-safe)
    dt: float  # seconds of video this sample represents
    tracks: List[TrackObservation]
    counts: Dict[str, int]  # persons / machinery / vehicle / cones in frame
    unattributed: List[Detection]  # NO-* detections that matched no person
    detections: List[Detection]  # raw detections (debug endpoint)


@dataclass
class ViolationEvent:
    event_id: str
    track_id: int
    item: str
    type: str
    t_start: float
    t_end: float
    frame_start: int
    frame_end: int
    peak_conf: float
    mean_conf: float
    samples: int
    near_hazard: bool = False
    evidence: Optional[dict] = None  # {"t", "frame_idx", "bbox"} of the best observation

    @property
    def duration(self) -> float:
        return max(0.0, self.t_end - self.t_start)
