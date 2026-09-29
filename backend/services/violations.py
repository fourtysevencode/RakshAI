"""Turn per-worker PPE observations into violation events.

One state machine per (track, PPE item), fed online during the frame loop so the
caller can capture evidence frames while they are still in memory:

- A run starts on the first `missing` observation and becomes an event once
  `missing` has persisted for >= min_violation_s (brief model flicker never
  becomes an event).
- A `present` observation discards a run that hasn't become an event yet.
- A run/event ends once no `missing` has been seen for > cooldown_s, so short
  occlusions (`unseen`) or single-frame flickers don't split one event in two.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from backend.models.schemas import VIOLATION_TYPE, FrameSample, ViolationEvent


@dataclass
class Run:
    run_id: str
    track_id: int
    item: str
    t_start: float
    frame_start: int
    last_t: float
    last_frame: int
    last_dt: float
    confs: List[float] = field(default_factory=list)
    best_conf: float = -1.0
    best_ref: Optional[dict] = None
    near: bool = False
    opened: bool = False

    def to_event(self) -> ViolationEvent:
        return ViolationEvent(
            event_id=self.run_id,
            track_id=self.track_id,
            item=self.item,
            type=VIOLATION_TYPE[self.item],
            t_start=self.t_start,
            t_end=self.last_t + self.last_dt,
            frame_start=self.frame_start,
            frame_end=self.last_frame,
            peak_conf=max(self.confs),
            mean_conf=sum(self.confs) / len(self.confs),
            samples=len(self.confs),
            near_hazard=self.near,
            evidence=self.best_ref,
        )


class ViolationTracker:
    def __init__(self, required_items: Iterable[str], min_violation_s: float = 1.0, cooldown_s: float = 2.0):
        self.required = [i for i in VIOLATION_TYPE if i in set(required_items)]
        self.min_violation_s = max(0.0, float(min_violation_s))
        self.cooldown_s = max(0.0, float(cooldown_s))
        self.events: List[ViolationEvent] = []
        self._runs: Dict[Tuple[int, str], Run] = {}
        self._counter = 0

    def feed(self, sample: FrameSample) -> Tuple[List[Tuple[Run, object]], List[str]]:
        """Process one sampled frame.

        Returns (improved, discarded): runs whose best evidence observation is
        in this frame (with that TrackObservation), and ids of runs dropped
        without becoming an event (their evidence can be freed).
        """
        improved: List[Tuple[Run, object]] = []
        discarded: List[str] = []
        t = sample.t
        for obs in sample.tracks:
            for item in self.required:
                key = (obs.track_id, item)
                run = self._runs.get(key)
                if run is not None and t - run.last_t > self.cooldown_s:
                    self._finish(key, discarded)
                    run = None
                po = obs.ppe[item]
                if po.state == "missing":
                    if run is None:
                        self._counter += 1
                        run = Run(
                            run_id=f"ev{self._counter:04d}", track_id=obs.track_id, item=item,
                            t_start=t, frame_start=sample.frame_idx,
                            last_t=t, last_frame=sample.frame_idx, last_dt=sample.dt,
                        )
                        self._runs[key] = run
                    run.last_t, run.last_frame, run.last_dt = t, sample.frame_idx, sample.dt
                    run.confs.append(po.confidence)
                    run.near = run.near or bool(obs.near)
                    if po.confidence > run.best_conf:
                        run.best_conf = po.confidence
                        run.best_ref = {"t": t, "frame_idx": sample.frame_idx, "bbox": obs.bbox, "item_bbox": po.bbox}
                        improved.append((run, obs))
                    if not run.opened and t - run.t_start >= self.min_violation_s - 1e-6:
                        run.opened = True
                elif po.state == "present" and run is not None and not run.opened:
                    self._finish(key, discarded)
        return improved, discarded

    def finalize(self) -> List[str]:
        """Close everything still open. Returns ids of discarded runs."""
        discarded: List[str] = []
        for key in list(self._runs):
            self._finish(key, discarded)
        self.events.sort(key=lambda e: (e.t_start, e.track_id, e.item))
        return discarded

    def _finish(self, key, discarded: List[str]) -> None:
        run = self._runs.pop(key)
        if run.opened:
            self.events.append(run.to_event())
        else:
            discarded.append(run.run_id)
