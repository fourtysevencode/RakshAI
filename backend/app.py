import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.models.schemas import PPE_ITEMS, VIOLATION_TYPE
from backend.services.pipeline import AnalysisSettings, Cancelled, run_analysis
from backend.utils.pdf_generation import generate_report
from backend.utils.predict import model_info

# Base directories
BASE_DIR = Path(__file__).resolve().parents[1]  # RakshAI/ (repo root)
STORAGE_DIR = BASE_DIR / "storage"
REPORTS_DIR = STORAGE_DIR / "reports"
FRONTEND_DIR = BASE_DIR / "frontend_static"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("RakshAI.Backend")

app = FastAPI(title="RakshAI Backend API")
# Allow the separately hosted frontend (Vercel production + preview URLs) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=os.environ.get("RAKSHAI_CORS_ORIGIN_REGEX", r"https://(raksh-ai[a-z0-9-]*\.vercel\.app|rakshai\.ronakbuilds\.tech)"),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# One analysis at a time: inference is CPU-bound, so parallel jobs only slow each
# other down (and the server). Further uploads wait in the queue.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")
_lock = threading.Lock()
_queue: List[str] = []

ANALYSIS_FPS_CHOICES = (0.0, 2.0, 5.0, 10.0)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class Job:
    def __init__(self, analysis_id: str, site_name: str, camera_id: str, filename: str,
                 settings: AnalysisSettings, temp_path: Path):
        self.analysis_id = analysis_id
        self.site_name = site_name
        self.camera_id = camera_id
        self.filename = filename
        self.settings = settings
        self.temp_path: Optional[Path] = temp_path  # uploaded video; deleted after processing
        self.status = "queued"  # queued | processing | completed | failed | cancelled
        self.stage = "queued"  # queued | detecting | report | done
        self.progress = 0
        self.frames_processed = 0
        self.video_position_s = 0.0
        self.duration_s = 0.0
        self.eta_seconds: Optional[float] = None
        self.created_at = _now_iso()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.error: Optional[str] = None
        self.cancel_requested = False
        self.result: Optional[dict] = None
        self.frames: List[dict] = []
        self.report_path: Optional[Path] = None

    def status_dict(self) -> dict:
        with _lock:
            position = _queue.index(self.analysis_id) + 1 if self.analysis_id in _queue else None
        return {
            "analysis_id": self.analysis_id,
            "site_name": self.site_name,
            "camera_id": self.camera_id,
            "filename": self.filename,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "queue_position": position,
            "frames_processed": self.frames_processed,
            "video_position_s": round(self.video_position_s, 2),
            "duration_s": round(self.duration_s, 2),
            "eta_seconds": None if self.eta_seconds is None else round(self.eta_seconds, 1),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "has_report": self.report_path is not None,
        }


analyses: Dict[str, Job] = {}


def _cleanup_upload(job: Job) -> None:
    if job.temp_path is not None:
        job.temp_path.unlink(missing_ok=True)
        job.temp_path = None


def _run_job(analysis_id: str) -> None:
    job = analyses[analysis_id]
    with _lock:
        if analysis_id in _queue:
            _queue.remove(analysis_id)
    if job.cancel_requested:
        job.status, job.stage, job.completed_at = "cancelled", "done", _now_iso()
        _cleanup_upload(job)
        return

    job.status, job.stage, job.started_at = "processing", "detecting", _now_iso()
    output = None
    try:
        from backend.services.video import probe
        job.duration_s = probe(job.temp_path).duration

        def on_progress(frames: int, t: float, elapsed: float) -> None:
            job.frames_processed = frames
            job.video_position_s = t
            if job.duration_s > 0:
                frac = min(1.0, t / job.duration_s)
                job.progress = int(90 * frac)
                if frac > 0.02:
                    job.eta_seconds = elapsed / frac - elapsed

        output = run_analysis(
            job.temp_path, job.settings, on_progress,
            should_cancel=lambda: job.cancel_requested,
            log=lambda msg: logger.info(f"analysis {analysis_id}: {msg}"),
        )
        job.result, job.frames = output.result, output.frames
        job.stage, job.progress, job.eta_seconds = "report", 93, None

        report_path = REPORTS_DIR / f"{analysis_id}.pdf"
        generate_report(
            report_path,
            {"analysis_id": analysis_id, "site_name": job.site_name, "camera_id": job.camera_id,
             "filename": job.filename, "created_at": job.created_at},
            output.result,
            output.evidence,
        )
        job.report_path = report_path
        job.status, job.stage, job.progress = "completed", "done", 100
        logger.info(f"analysis {analysis_id} completed in {output.result.get('processing_s')}s")
    except Cancelled:
        job.status, job.stage = "cancelled", "done"
        logger.info(f"analysis {analysis_id} cancelled")
    except Exception as exc:
        logger.exception(f"analysis {analysis_id} failed")
        job.status, job.stage, job.error = "failed", "done", str(exc)
    finally:
        job.completed_at = _now_iso()
        job.eta_seconds = None
        if output is not None:
            output.evidence.clear()  # evidence images exist only for the PDF
        _cleanup_upload(job)


def _get_job(analysis_id: str) -> Job:
    job = analyses.get(analysis_id)
    if job is None:
        raise HTTPException(status_code=404, detail="analysis_id not found")
    return job


def _get_result(analysis_id: str) -> dict:
    job = _get_job(analysis_id)
    if job.result is None:
        raise HTTPException(status_code=409, detail=f"analysis is {job.status}; results not available")
    return job.result


@app.get("/", response_class=HTMLResponse)
def frontend_home():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return "<html><body><h1>RakshAI Backend</h1><p>Frontend not configured.</p></body></html>"


@app.get("/api/v1/health")
def health():
    info = model_info()
    with _lock:
        queued = len(_queue)
    running = sum(1 for j in analyses.values() if j.status == "processing")
    return {
        "status": "ok" if info["loaded"] else "degraded",
        "model": info,
        "ppe_items": list(PPE_ITEMS),
        "violation_types": [VIOLATION_TYPE[i] for i in PPE_ITEMS],
        "analysis_fps_choices": list(ANALYSIS_FPS_CHOICES),
        "queue": {"running": running, "queued": queued},
    }


@app.post("/api/v1/analysis")
def create_analysis(
    video: UploadFile = File(...),
    site_name: str = Form(...),
    camera_id: str = Form(...),
    analysis_fps: float = Form(5.0),
    required_ppe: str = Form(",".join(PPE_ITEMS)),
    conf_threshold: float = Form(0.25),
    min_violation_s: float = Form(1.0),
    debug: bool = Form(False),
):
    required = [p.strip().lower() for p in required_ppe.split(",") if p.strip()]
    unknown = [p for p in required if p not in PPE_ITEMS]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown PPE item(s): {unknown}; choose from {list(PPE_ITEMS)}")
    if not 0 <= analysis_fps <= 30:
        raise HTTPException(status_code=422, detail="analysis_fps must be between 0 (every frame) and 30")
    if not 0.05 <= conf_threshold <= 0.95:
        raise HTTPException(status_code=422, detail="conf_threshold must be between 0.05 and 0.95")
    if not 0 <= min_violation_s <= 30:
        raise HTTPException(status_code=422, detail="min_violation_s must be between 0 and 30")

    # Uploaded videos are never persisted: copy to a temp file (OpenCV needs a path),
    # process it, then delete it when the job finishes.
    suffix = Path(video.filename or "").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="rakshai_") as tmp:
        shutil.copyfileobj(video.file, tmp, length=1024 * 1024)
        tmp_path = Path(tmp.name)

    analysis_id = uuid.uuid4().hex
    settings = AnalysisSettings(
        analysis_fps=analysis_fps, required_ppe=required, conf_threshold=conf_threshold,
        min_violation_s=min_violation_s, debug=debug,
    )
    job = Job(analysis_id, site_name.strip(), camera_id.strip(), video.filename or "video", settings, tmp_path)
    analyses[analysis_id] = job
    with _lock:
        _queue.append(analysis_id)
    try:
        _executor.submit(_run_job, analysis_id)
    except Exception:
        with _lock:
            _queue.remove(analysis_id)
        _cleanup_upload(job)
        del analyses[analysis_id]
        raise HTTPException(status_code=500, detail="could not schedule analysis")
    return job.status_dict()


@app.get("/api/v1/analysis/{analysis_id}")
def get_analysis_status(analysis_id: str):
    return _get_job(analysis_id).status_dict()


@app.delete("/api/v1/analysis/{analysis_id}")
def cancel_analysis(analysis_id: str):
    job = _get_job(analysis_id)
    if job.status in ("queued", "processing"):
        job.cancel_requested = True
    return job.status_dict()


@app.get("/api/v1/analysis/{analysis_id}/results")
def get_analysis_results(analysis_id: str):
    r = _get_result(analysis_id)
    keys = ("summary", "ppe", "timeline", "occupancy", "scene", "findings", "video", "settings", "model", "processing_s")
    return {"analysis_id": analysis_id, **{k: r.get(k) for k in keys}}


@app.get("/api/v1/analysis/{analysis_id}/persons")
def get_analysis_persons(analysis_id: str):
    return {"analysis_id": analysis_id, "persons": _get_result(analysis_id)["persons"]}


@app.get("/api/v1/analysis/{analysis_id}/events")
def get_analysis_events(analysis_id: str, type: Optional[str] = None, worker_id: Optional[int] = None):
    events = _get_result(analysis_id)["events"]
    if type:
        events = [e for e in events if e["type"] == type.upper()]
    if worker_id is not None:
        events = [e for e in events if e["worker_id"] == worker_id]
    return {"analysis_id": analysis_id, "events": events}


@app.get("/api/v1/analysis/{analysis_id}/tracks")
def get_analysis_tracks(analysis_id: str):
    """Per-worker boxes over time (normalized 0-1): rows are [t, x1, y1, x2, y2, active_violation_types]."""
    return {"analysis_id": analysis_id, "tracks": _get_result(analysis_id)["tracks"]}


@app.get("/api/v1/analysis/{analysis_id}/frames")
def get_analysis_frames(analysis_id: str, start: float = 0.0, end: Optional[float] = None, limit: int = 200):
    """Raw per-sample detections (debug). Boxes are pixels of the processed frame."""
    _get_result(analysis_id)
    frames = [f for f in analyses[analysis_id].frames if f["t"] >= start and (end is None or f["t"] <= end)]
    return {"analysis_id": analysis_id, "total": len(frames), "frames": frames[:max(1, min(limit, 5000))]}


@app.get("/api/v1/analysis/{analysis_id}/report")
def get_analysis_report(analysis_id: str):
    job = _get_job(analysis_id)
    if not job.report_path or not job.report_path.exists():
        raise HTTPException(status_code=404, detail="report not generated yet")
    safe_site = "".join(c if c.isalnum() else "-" for c in job.site_name)[:40] or "site"
    return FileResponse(job.report_path, media_type="application/pdf",
                        filename=f"RakshAI_{safe_site}_{analysis_id[:8]}.pdf")


# Static assets for the vanilla frontend (app.js etc.)
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
