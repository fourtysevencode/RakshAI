# RakshAI

RakshAI is an AI-powered construction-site safety analysis system that analyzes CCTV or other video footage for PPE compliance violations and generates a structured safety report in PDF format.

The core idea is to turn frame-level computer-vision detections into meaningful, deduplicated safety events that can be reviewed by site safety personnel.

## Concept

```text
Video Upload
     ↓
Analysis Job
     ↓
Video Processing (OpenCV)
     ↓
PPE Detection (YOLO + best.onnx)
     ↓
Worker Tracking
     ↓
Violation Event Detection
     ↓
Analytics
     ↓
Evidence Frames + Charts
     ↓
PDF Safety Report
```

A key design principle is that raw model detections should not directly be treated as violations. The application layer should associate PPE detections with individual workers, track those workers over time, and convert persistent missing-PPE observations into deduplicated violation events.

## Current Stack

- **Backend:** FastAPI (single-worker job queue, in-memory results)
- **Computer Vision:** OpenCV
- **Object Detection:** Ultralytics YOLO11n using `models/best.onnx` (10 classes: `Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, vehicle`)
- **Inference:** Ultralytics, with a plain ONNX Runtime fallback
- **Worker tracking:** ByteTrack (Ultralytics' tracker, fed with our own person detections)
- **Reporting:** ReportLab + matplotlib (charts, worker profiles, in-memory evidence frames)
- **Frontend:** vanilla HTML + Tailwind (Play CDN) + JS in `frontend_static/`

## Repository Structure

```text
RakshAI/
├── backend/
│   ├── app.py                  # FastAPI app, job queue, API routes
│   ├── models/schemas.py       # dataclasses + PPE/class mappings
│   ├── services/
│   │   ├── video.py            # probing + timestamp-based frame sampling (VFR-safe)
│   │   ├── tracking.py         # ByteTrack wrapper, duplicate person suppression
│   │   ├── association.py      # PPE / NO-PPE boxes -> tracked worker
│   │   ├── violations.py       # per-worker persistence/cooldown state machine
│   │   ├── hazards.py          # worker <-> machinery/vehicle proximity (2D)
│   │   ├── evidence.py         # in-memory thumbnails + annotated event frames
│   │   ├── analytics.py        # per-worker summaries, PPE breakdown, timeline, findings
│   │   └── pipeline.py         # runs the whole analysis for one video
│   ├── utils/
│   │   ├── predict.py          # model loading + predict_frame()
│   │   └── pdf_generation.py   # PDF report
│   ├── tests/                  # pytest unit tests (no model needed)
│   └── requirements.txt
├── frontend_static/            # index.html + app.js (served at / and /frontend)
├── models/best.onnx
└── storage/
    ├── uploads/                # legacy sample clips only; new uploads are never stored
    └── reports/                # generated PDFs
```

Uploaded videos are written to a temporary file, analysed, and deleted. Evidence images only exist in memory while the PDF is built. The web UI reviews footage from the user's local copy of the video.

A single-worker thread pool runs one analysis at a time; further uploads wait in a queue. Redis/Celery can be introduced later if concurrent analysis jobs become necessary.

## API Design

All routes are under `/api/v1`.

| Method & path | Purpose |
|---|---|
| `GET /health` | Model status, class list, queue length |
| `POST /analysis` | Upload a video and queue an analysis |
| `GET /analysis/{id}` | Status: `status`, `stage`, `progress`, `queue_position`, `frames_processed`, `eta_seconds`, `error` |
| `DELETE /analysis/{id}` | Cancel a queued or running analysis |
| `GET /analysis/{id}/results` | Summary, PPE breakdown, timeline buckets, occupancy, scene context, key findings |
| `GET /analysis/{id}/persons` | Per-worker summaries |
| `GET /analysis/{id}/events` | Violation events (filter with `?type=NO_MASK&worker_id=2`) |
| `GET /analysis/{id}/tracks` | Per-worker boxes over time for video overlays: rows of `[t, x1, y1, x2, y2, active_types]`, 0–1 normalized |
| `GET /analysis/{id}/frames?start=&end=&limit=` | Raw per-frame detections (debugging) |
| `GET /analysis/{id}/report` | PDF report |

`POST /analysis` form fields:

```text
video            <file>                  required
site_name        Construction Site A     required
camera_id        CAM-03                  required
analysis_fps     5                       2 | 5 | 10 | 0 (= every frame)
required_ppe     hardhat,vest,mask       which items count as violations
conf_threshold   0.25
min_violation_s  1.0                     shorter lapses are reported as "brief"
debug            false                   per-frame server logs
```

Statuses: `queued`, `processing`, `completed`, `failed`, `cancelled`.

Example person (`/persons`):

```json
{
  "worker_id": 1,
  "label": "Worker 1",
  "first_seen": 0.0,
  "last_seen": 15.9,
  "visible_s": 12.1,
  "status": "non_compliant",
  "severity": {"score": 31.6, "level": "high"},
  "violation_s": 12.0,
  "ppe": {
    "hardhat": {"required": true, "status": "violation", "compliance": 0.43, "present_s": 5.0, "missing_s": 6.6, "unseen_s": 0.5, "events": 1},
    "vest": {"required": true, "status": "ok", "compliance": 1.0}
  },
  "exposure": {"machinery": 2.4, "vehicle": 0.0},
  "best_frame": {"t": 11.0, "bbox": [0.58, 0.07, 0.83, 0.93]}
}
```

Worker status is `non_compliant` (at least one violation event), `compliant` (required PPE seen being worn), or `unconfirmed` (PPE never seen clearly enough). Per-item status is `violation`, `brief` (missing, but shorter than `min_violation_s`), `ok`, or `unseen`.

Example event (`/events`):

```json
{
  "event_id": "ev0023",
  "worker_id": 1,
  "type": "NO_HARDHAT",
  "t_start": 10.81,
  "t_end": 15.5,
  "duration": 4.69,
  "peak_conf": 0.8,
  "near_hazard": true,
  "severity": 21.1
}
```

## Processing Pipeline

### 1. Video service

Responsible for:

- Saving uploaded videos
- Reading video metadata
- Getting FPS, frame count and duration
- Sampling frames at a configurable analysis FPS
- Extracting evidence frames

Useful functions:

```python
get_video_metadata(path)
iter_video_frames(path, target_fps=5)
extract_frame(path, timestamp)
save_frame(frame, path)
```

A starting point of around **5 FPS** can significantly reduce inference cost compared with processing every frame of a 25/30 FPS CCTV recording. This should be benchmarked against detection accuracy for the target footage.

### 2. Inference service

`best.onnx` should be loaded once when the application/worker starts rather than once per request.

The inference layer should expose an application-independent interface such as:

```python
predict_frame(frame)
normalize_detections(results)
```

Normalize model output into a common detection representation:

```text
Detection
├── class_id
├── class_name
├── confidence
└── bbox [x1, y1, x2, y2]
```

This keeps the rest of RakshAI independent from the exact model implementation.

### 3. Worker tracking

Frame-by-frame detection is not enough for useful violation counts. A worker visible without a helmet for 100 consecutive frames should generally represent one continuous event, not 100 separate violations.

Use a tracking layer, such as ByteTrack through the Ultralytics pipeline, to maintain worker identities:

```text
Worker #17
├── first_seen: 83.2s
├── last_seen: 94.8s
├── helmet_present: false
└── vest_present: true
```

Useful functions:

```python
update_tracks(detections)
get_active_workers()
close_stale_tracks()
```

### 4. Violation service

The violation service converts detections + worker tracking into safety events.

Example:

```text
Person #17
├── helmet: false
├── vest: true
└── boots: true
```

becomes:

```text
NO_HELMET
start: 83.2s
end: 88.7s
duration: 5.5s
track_id: 17
```

Useful functions:

```python
associate_ppe_with_workers()
detect_missing_ppe()
start_violation()
update_violation()
close_violation()
deduplicate_events()
```

A cooldown/persistence threshold should be used to prevent brief model flicker from generating excessive events.

## Violation Event Model

A normalized event should contain enough information for analytics and evidence generation:

```python
@dataclass
class ViolationEvent:
    event_id: str
    timestamp_start: float
    timestamp_end: float
    duration: float
    track_id: int
    violation_type: str
    confidence: float
    frame_start: int
    frame_end: int
```

## Analytics

The analytics layer should calculate more than a raw violation count.

### Basic metrics

- Total workers detected
- Total violation events
- Unique workers involved in violations
- Average violation duration
- Longest violation
- Analysis duration

### PPE breakdown

```text
No helmet       31
No safety vest  17
No boots         4
```

### Rates

- Violations per worker
- Violations per hour
- PPE-specific violation rate
- Percentage of observed worker-time associated with violations

Raw counts need context. For example, 100 violations across 5 workers for one hour is not directly comparable to 100 violations across 50 workers for ten hours.

### Timeline analysis

Aggregate events into time buckets to identify periods with higher numbers of observed violations.

```text
08:00 → 3
09:00 → 12
10:00 → 6
11:00 → 4
```

This can be represented as a line/bar chart in the report.

## Evidence Frames

Each violation event keeps its highest-confidence frame, annotated with the worker box and the missing item (up to the 60 most severe events), plus one thumbnail per worker. These images are held **in memory only**, embedded in the PDF and then discarded. Nothing is written to `storage/processed/`.

## PDF Report

The report should contain:

1. **Executive summary**
   - Site
   - Camera
   - Analysis date
   - Video duration
   - Workers observed
   - Total PPE violations
   - Violation rate
   - Average event duration

2. **PPE breakdown**
   - No helmet
   - No safety vest
   - No boots
   - Other model-supported PPE categories

3. **Violations over time**
   - Timeline chart
   - Time buckets with event counts

4. **Violation events**
   - Timestamp
   - Violation type
   - Duration
   - Confidence
   - Track ID where appropriate

5. **Evidence**
   - Representative frames for significant events

6. **Methodology**
   - Model used
   - Sampling FPS
   - Detection threshold
   - Processing information

The report should clearly communicate that AI-generated observations are subject to model limitations and should be reviewed by qualified safety personnel before being used for compliance decisions.

## Database

For the MVP, SQLite is sufficient.

### `analyses`

```text
id
filename
site_name
camera_id
duration
status
created_at
completed_at
```

### `violations`

```text
id
analysis_id
track_id
type
start_time
end_time
duration
confidence
evidence_path
```

PostgreSQL can replace SQLite later without changing the core processing architecture.

## Recommended Development Phases

### Phase 1 — Functional MVP

```text
Upload video
→ process video
→ YOLO inference
→ detect PPE
→ count violations
→ generate PDF
```

### Phase 2 — Reliable event detection

Add:

- Worker tracking
- Violation duration
- Evidence frames
- Timeline analytics
- Charts
- Event deduplication

### Phase 3 — Production-style application

Add:

- Asynchronous analysis jobs
- Progress reporting
- Persistent database
- Frontend dashboard
- Analysis history
- Downloadable reports

### Phase 4 — Advanced monitoring

Potential future features:

- Multiple CCTV cameras
- Live CCTV streams
- Real-time safety alerts
- Historical site analytics
- Worker-level trends where appropriate
- Configurable site-specific PPE rules

## Engineering Priorities

The most important part of RakshAI is the **detection → tracking → event** layer. The object detector provides raw computer-vision observations, but the application needs to transform those observations into consistent, deduplicated and explainable safety events.

Keep inference, tracking, violation logic, analytics and report generation as separate services so each component can be tested and replaced independently.

Also use an absolute model path based on the project location rather than relying on the current working directory:

```python
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "best.onnx"

model = YOLO(str(MODEL_PATH))
```

This avoids failures caused by launching the application from a different working directory.

## Goal

RakshAI aims to provide a practical way to analyze construction-site footage, quantify observed PPE safety violations, surface when and how violations occur, and provide evidence-backed PDF reports for human review.

## Quickstart

- Python 3.9+. From the repo root:
  - `python -m venv venv && source venv/bin/activate` (optional)
  - `pip install -r backend/requirements.txt`
- The model must be at `models/best.onnx`.
- Run: `./start_all.sh` (or `uvicorn backend.app:app --port 8000`), then open http://localhost:8000.
- Tests: `python -m pytest backend/tests`
- Notes:
  - Analyses and results live in memory and are lost on restart; PDFs in `storage/reports/` remain.
  - If the model can't be loaded, analyses fail with an error rather than reporting "no detections".
