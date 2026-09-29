"""Full analysis of one video: sample -> detect -> track -> associate -> violations -> analytics."""
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from backend.models.schemas import (
    CONE_CLASS, PERSON_CLASS, PPE_ITEMS, Detection, FrameSample,
)
from backend.services.analytics import build_results, severity_score
from backend.services.association import associate
from backend.services.evidence import MAX_EVENT_IMAGES, EvidenceStore
from backend.services.hazards import nearby_hazards
from backend.services.tracking import PersonTracker, suppress_duplicate_persons
from backend.services.video import VideoInfo, iter_sampled_frames, probe
from backend.services.violations import ViolationTracker
from backend.utils.predict import model_info, predict_frame

# Person boxes down to this confidence go to the tracker (ByteTrack's low-score pass)
TRACK_CONF = 0.1


class Cancelled(Exception):
    pass


@dataclass
class AnalysisSettings:
    analysis_fps: float = 5.0  # 0 = every frame
    required_ppe: List[str] = field(default_factory=lambda: list(PPE_ITEMS))
    conf_threshold: float = 0.25
    min_violation_s: float = 1.0
    cooldown_s: float = 2.0
    debug: bool = False


@dataclass
class AnalysisOutput:
    result: dict
    frames: List[dict]  # raw per-sample detections, for the debug endpoint
    evidence: EvidenceStore
    info: VideoInfo


def run_analysis(
    path: Path,
    settings: AnalysisSettings,
    on_progress: Optional[Callable[[int, float, float], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> AnalysisOutput:
    info = probe(path)
    if settings.analysis_fps and settings.analysis_fps > 0:
        sample_fps = float(settings.analysis_fps)
    else:
        sample_fps = info.fps if 0 < info.fps <= 120 else 30.0
    nominal_dt = 1.0 / sample_fps

    tracker = PersonTracker(sample_fps)
    violations = ViolationTracker(settings.required_ppe, settings.min_violation_s, settings.cooldown_s)
    evidence = EvidenceStore()
    samples: List[FrameSample] = []
    frames_debug: List[dict] = []
    prev_t: Optional[float] = None
    started = time.time()

    try:
        for frame_idx, t, frame in iter_sampled_frames(path, settings.analysis_fps):
            if should_cancel and should_cancel():
                raise Cancelled()
            raw = [Detection.from_dict(d) for d in predict_frame(frame, conf=min(TRACK_CONF, settings.conf_threshold))]
            persons = suppress_duplicate_persons([d for d in raw if d.class_name == PERSON_CLASS])
            others = [d for d in raw if d.class_name != PERSON_CLASS and d.confidence >= settings.conf_threshold]
            tracked = tracker.update(persons)
            observations, unattributed = associate(tracked, others)
            for o in observations:
                o.near = nearby_hazards(o.bbox, others)

            # In "every frame" mode frames can be irregular (VFR), so use the real gap
            dt = nominal_dt
            if not settings.analysis_fps and prev_t is not None and t > prev_t:
                dt = min(t - prev_t, 4 * nominal_dt)
            prev_t = t
            sample = FrameSample(
                frame_idx=frame_idx, t=t, dt=dt, tracks=observations,
                counts={
                    "persons": len(tracked),
                    "machinery": sum(1 for d in others if d.class_name == "machinery"),
                    "vehicle": sum(1 for d in others if d.class_name == "vehicle"),
                    "cones": sum(1 for d in others if d.class_name == CONE_CLASS),
                },
                unattributed=unattributed,
                detections=[d for d in raw if d.class_name != PERSON_CLASS or d.confidence >= settings.conf_threshold],
            )
            samples.append(sample)
            frames_debug.append({
                "frame_idx": frame_idx,
                "t": round(t, 3),
                "detections": [asdict(d) for d in sample.detections],
                "tracks": [{"track_id": o.track_id, "bbox": [round(v, 1) for v in o.bbox]} for o in observations],
            })

            for o in observations:
                evidence.consider_thumbnail(o.track_id, o.confidence, frame, o.bbox)
            improved, discarded = violations.feed(sample)
            for run, obs in improved:
                evidence.set_event_image(run.run_id, frame, obs.bbox, obs.ppe[run.item].bbox, run.item)
            evidence.drop(discarded)

            if log and settings.debug:
                log(f"t={t:.2f}s frame={frame_idx} tracks={[o.track_id for o in observations]} "
                    f"dets={[d.class_name for d in raw]}")
            if on_progress:
                on_progress(len(samples), t, time.time() - started)

        evidence.drop(violations.finalize())
        # Keep evidence for the most severe events only
        top = sorted(violations.events, key=severity_score, reverse=True)[:MAX_EVENT_IMAGES]
        evidence.keep_only([e.event_id for e in top])

        result = build_results(
            samples, violations.events, info.duration, (info.proc_width, info.proc_height), settings.required_ppe,
        )
    except BaseException:
        evidence.clear()
        raise

    result["video"] = {
        "duration_s": round(info.duration, 3),
        "width": info.width,
        "height": info.height,
        "reported_fps": round(info.fps, 3),
        "processed_width": info.proc_width,
        "processed_height": info.proc_height,
    }
    result["settings"] = {**asdict(settings), "sample_fps": round(sample_fps, 3)}
    result["model"] = model_info()
    result["processing_s"] = round(time.time() - started, 2)
    return AnalysisOutput(result=result, frames=frames_debug, evidence=evidence, info=info)
