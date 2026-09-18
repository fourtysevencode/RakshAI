from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import datetime
# OpenCV import is guarded to avoid startup crashes if system GL libraries
# are unavailable in the execution environment. The headless variant should be
# preferred (handled via dependencies) to avoid GUI dependencies like libGL.
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore
import uuid
import numpy as np
import logging
from typing import List, Optional

# Helper to coerce boolean form values that may come in as strings (e.g., checkbox "on").
def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "y", "on"):  # common truthy literals
            return True
        if v in ("0", "false", "no", "n", "off"):  # common falsy literals
            return False
    return bool(value)

from backend.utils.predict import predict_frame
from backend.utils.pdf_generation import generate_pdf

# Helper: downscale video to 1280x720 (maintaining aspect ratio) if larger
def _downscale_video_to_720p(input_path: Path) -> tuple[Path, bool]:
    # If OpenCV is not available, skip processing
    if cv2 is None:
        return input_path, False
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            cap.release()
            return input_path, False
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()

        # If already within 1280x720, nothing to do
        if w <= 1280 and h <= 720:
            return input_path, False

        scale = min(1280.0 / w, 720.0 / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        out_path = input_path.with_name(input_path.stem + "_720p" + input_path.suffix)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cap2 = cv2.VideoCapture(str(input_path))
        if not cap2.isOpened():
            cap2.release()
            return input_path, False
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (new_w, new_h))
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            out.write(resized)
        cap2.release()
        out.release()
        return out_path, True
    except Exception:
        # On any failure, return original path to avoid masking real failures
        return input_path, False

# Base directories
BASE_DIR = Path(__file__).resolve().parents[1]  # RakshAI/ (repo root)
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
PROCESSED_DIR = STORAGE_DIR / "processed"
REPORTS_DIR = STORAGE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best.onnx"

# Ensure required directories exist
for d in [UPLOADS_DIR, PROCESSED_DIR, REPORTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RakshAI Backend API (Demo MVP)")
logger = logging.getLogger("RakshAI.Backend")

# In-memory store for analyses (simple MVP state machine)
analyses = {}


def _now_iso():
    from datetime import datetime as _dt
    return _dt.utcnow().isoformat() + "Z"


class AnalysisSummary:
    def __init__(self, analysis_id: str, site_name: str, camera_id: str, video_path: str, frame_skip: int = 1):
        self.analysis_id = analysis_id
        self.site_name = site_name
        self.camera_id = camera_id
        self.video_path = video_path
        self.status = "queued"  # queued | processing | completed | failed
        self.progress = 0
        self.created_at = _now_iso()
        self.completed_at: Optional[str] = None
        self.results: Optional[dict] = None
        self.events: List[dict] = []
        self.report_path: Optional[str] = None
        self.frame_skip = int(frame_skip) if frame_skip is not None else 1
        self.process_all_frames = False
        self.debug_all_frames = False

        self.temp_video_path = None  # Path to temporary uploaded video (if any)

    def as_dict(self):
        return {
            "analysis_id": self.analysis_id,
            "site_name": self.site_name,
            "camera_id": self.camera_id,
            "video_path": str(self.video_path),
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "results": self.results,
            "events": self.events,
            "report_path": str(self.report_path) if self.report_path else None,
            "frame_skip": self.frame_skip,
            "process_all_frames": self.process_all_frames,
            "debug_all_frames": self.debug_all_frames,
        }


@app.get("/", response_class=HTMLResponse)
def frontend_home():
    # Serve a simple vanilla HTML frontend from frontend_static/index.html
    index_path = BASE_DIR / "frontend_static" / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<html><body><h1>RakshAI Backend</h1><p>Frontend not configured.</p></body></html>"


@app.post("/api/v1/analysis")
async def create_analysis(
    video: UploadFile = File(...),
    site_name: str = Form(...),
    camera_id: str = Form(...),
    frame_skip: int = Form(1),
    process_all_frames: Optional[str] = Form(None),
    debug_all_frames: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
):
    # Do not persist uploaded videos. Write to a temporary file, process, then clean up.
    video_ext = Path(video.filename).suffix or ".mp4"
    analysis_id = uuid.uuid4().hex
    import tempfile
    content = await video.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=video_ext) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    # Downscale to 720p if needed for processing (temporary path may be used)
    video_path, downscaled = _downscale_video_to_720p(tmp_path)

    summary = AnalysisSummary(analysis_id, site_name, camera_id, video_path, frame_skip=frame_skip)
    summary.process_all_frames = _to_bool(process_all_frames)
    summary.debug_all_frames = _to_bool(debug_all_frames)
    # Record the temp file so we can delete it later
    summary.temp_video_path = tmp_path
    analyses[analysis_id] = summary

    # Schedule background processing
    if background_tasks is not None:
        background_tasks.add_task(_process_analysis, analysis_id)

    summary.status = "queued"
    summary.progress = 0
    return {"analysis_id": analysis_id, "status": summary.status}


def _load_pdf_path(analysis_id: str) -> Path:
    return REPORTS_DIR / f"{analysis_id}.pdf"


async def _process_analysis(analysis_id: str):
    # Simple background worker that processes video frames and runs inference
    if analysis_id not in analyses:
        return
    analysis = analyses[analysis_id]
    analysis.status = "processing"
    analysis.progress = 5

    video_path = analysis.video_path
    if not video_path.exists():
        analysis.status = "failed"
        analysis.completed_at = _now_iso()
        return

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            analysis.status = "failed"
            analysis.completed_at = _now_iso()
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Determine interval based on frame_skip if provided
        if analysis.frame_skip and analysis.frame_skip > 1:
            interval = max(1, int(analysis.frame_skip))
        elif getattr(analysis, 'process_all_frames', False):
            interval = 1
        else:
            target_fps = 5
            interval = max(1, int(round(fps / target_fps)))

        detections_all = []
        frame_idx = 0
        frames_analyzed = 0
        frames_read = 0
        logger.info(f"Processing analysis {analysis_id}: fps={fps}, frame_count={frame_count}, interval={interval}, src={video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_read += 1
            if frame_idx % interval == 0:
                frames_analyzed += 1
                dets = predict_frame(frame)
                # Always log the frame with its detections (even if empty)
                detections_all.append({"frame": frame_idx, "detections": dets})
            if getattr(analysis, 'debug_all_frames', False):
                logger.info(f"analysis {analysis_id}: frame={frame_idx}, interval={interval}, detections={len(dets) if 'dets' in locals() else 0}")
            frame_idx += 1

        cap.release()
        frames_total = frames_read

        # Build a simple results payload
        total_duration = (frame_count / fps) if fps else 0.0
        if analysis.debug_all_frames:
            logger.info(
                f"analysis {analysis_id} finished: frames_read={frames_total}, frames_analyzed={frames_analyzed}, interval={interval}, duration={total_duration:.2f}s, fps={fps}"
            )
        results = {
            "duration_seconds": float(total_duration),
            "frames_processed": int(frames_analyzed),
            "detections": detections_all,
        }

        # Naive violations computation and logging (log per-frame missing PPE)
        violations_events = []
        types_counter = {"NO_HELMET": 0, "NO_VEST": 0, "NO_BOOTS": 0, "NO_MASK": 0}
        for fg in detections_all:
            fidx = fg.get("frame", 0)
            dets = fg.get("detections", [])
            t = float(fidx) / float(fps) if fps else fidx
            helmet_present = any(((d.get("class_name", "").lower() == "helmet") or (int(d.get("class_id", 0)) == 0)) for d in dets)
            vest_present = any(((d.get("class_name", "").lower() == "vest") or (int(d.get("class_id", 0)) == 1)) for d in dets)
            boots_present = any(((d.get("class_name", "").lower() == "boots") or (int(d.get("class_id", 0)) == 2)) for d in dets)
            mask_present = any(((d.get("class_name", "").lower() == "mask") or (int(d.get("class_id", 0)) == 3)) for d in dets)
            if not helmet_present:
                violations_events.append({"timestamp": t, "type": "NO_HELMET", "frame_start": fidx, "duration": 0.0, "log": f"Frame {fidx}: NO_HELMET"})
                types_counter["NO_HELMET"] += 1
            if not vest_present:
                violations_events.append({"timestamp": t, "type": "NO_VEST", "frame_start": fidx, "duration": 0.0, "log": f"Frame {fidx}: NO_VEST"})
                types_counter["NO_VEST"] += 1
            if not boots_present:
                violations_events.append({"timestamp": t, "type": "NO_BOOTS", "frame_start": fidx, "duration": 0.0, "log": f"Frame {fidx}: NO_BOOTS"})
                types_counter["NO_BOOTS"] += 1
            if not mask_present:
                violations_events.append({"timestamp": t, "type": "NO_MASK", "frame_start": fidx, "duration": 0.0, "log": f"Frame {fidx}: NO_MASK"})
                types_counter["NO_MASK"] += 1

        total_violations = len(violations_events)
        duration_sec = total_duration if total_duration > 0 else 1e-6
        violation_rate = total_violations / duration_sec
        results["violations"] = {
            "total_violations": total_violations,
            "violation_rate": violation_rate,
            "types": types_counter,
        }
        analysis.events = violations_events  # for PDF and UI log

        # Generate a PDF report (best-effort)
        pdf_path = _load_pdf_path(analysis_id)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        generate_pdf(analysis_id, {"site_name": analysis.site_name, "camera_id": analysis.camera_id}, results, violations_events, pdf_path)
        analysis.report_path = str(pdf_path)
        analysis.progress = 100
        analysis.status = "completed"
        analysis.completed_at = _now_iso()
    except Exception as exc:
        analysis.status = "failed"
        analysis.completed_at = _now_iso()
        analysis.results = {"error": str(exc)}
    finally:
        # Cleanup uploaded/temporary video files to avoid storage
        try:
            video_path = analysis.video_path
            paths_to_remove = []
            if isinstance(video_path, Path):
                paths_to_remove.append(video_path)
            # Also consider the 720p variant path if it exists
            try:
                p720 = video_path.with_name(video_path.stem + "_720p" + video_path.suffix)  # type: ignore
                if isinstance(p720, Path):
                    paths_to_remove.append(p720)
            except Exception:
                pass
            for p in paths_to_remove:
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass
        except Exception:
            pass
        # Also cleanup the original temporary uploaded video if present
        try:
            tmp = getattr(analysis, 'temp_video_path', None)
            if isinstance(tmp, Path) and tmp.exists():
                tmp.unlink()
        except Exception:
            pass


@app.get("/api/v1/analysis/{analysis_id}")
def get_analysis_status(analysis_id: str):
    if analysis_id not in analyses:
        raise HTTPException(status_code=404, detail="analysis_id not found")
    a = analyses[analysis_id]
    return a.as_dict()


@app.get("/api/v1/analysis/{analysis_id}/results")
def get_analysis_results(analysis_id: str):
    if analysis_id not in analyses:
        raise HTTPException(status_code=404, detail="analysis_id not found")
    a = analyses[analysis_id]
    return {"analysis_id": analysis_id, "results": a.results or {}}


@app.get("/api/v1/analysis/{analysis_id}/events")
def get_analysis_events(analysis_id: str):
    if analysis_id not in analyses:
        raise HTTPException(status_code=404, detail="analysis_id not found")
    a = analyses[analysis_id]
    return {"analysis_id": analysis_id, "events": a.events or []}


@app.get("/api/v1/analysis/{analysis_id}/report")
def get_analysis_report(analysis_id: str):
    if analysis_id not in analyses:
        raise HTTPException(status_code=404, detail="analysis_id not found")
    a = analyses[analysis_id]
    if not a.report_path:
        raise HTTPException(status_code=404, detail="report not generated yet")
    return FileResponse(a.report_path, media_type="application/pdf", filename=f"{analysis_id}.pdf")

# Serve the static frontend (vanilla HTML/JS/CSS) under /frontend
app.mount("/frontend", StaticFiles(directory=str(BASE_DIR / "frontend_static"), html=True), name="frontend")
