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

- **Backend:** FastAPI
- **Computer Vision:** OpenCV
- **Object Detection:** Ultralytics YOLO using `models/best.onnx`
- **Inference:** ONNX Runtime / Ultralytics
- **Data Processing:** NumPy, Pandas
- **Validation:** Pydantic
- **Reporting:** PDF generation with charts and evidence frames

## Repository Structure

The backend is intended to evolve toward the following structure:

```text
RakshAI/
├── backend/
│   ├── app.py
│   ├── api/
│   │   ├── routes_analysis.py
│   │   ├── routes_results.py
│   │   └── routes_health.py
│   ├── services/
│   │   ├── video_service.py
│   │   ├── inference_service.py
│   │   ├── tracking_service.py
│   │   ├── violation_service.py
│   │   ├── analytics_service.py
│   │   └── report_service.py
│   ├── models/
│   │   ├── schemas.py
│   │   └── enums.py
│   ├── utils/
│   │   ├── predict.py
│   │   ├── video.py
│   │   ├── timestamps.py
│   │   └── pdf_generation.py
│   ├── workers/
│   │   └── analysis_worker.py
│   └── config.py
├── models/
│   └── best.onnx
├── storage/
│   ├── uploads/
│   ├── processed/
│   └── reports/
└── requirements.txt
```

For the MVP, this structure does not require a distributed queue. A FastAPI background task or separate worker process is sufficient. Redis/Celery can be introduced later if concurrent analysis jobs become necessary.

## API Design

### Create an analysis

```http
POST /api/v1/analysis
```

Multipart form data:

```text
video: <video file>
site_name: Construction Site A
camera_id: CAM-03
```

Response:

```json
{
  "analysis_id": "a83d1f",
  "status": "queued"
}
```

### Check analysis status

```http
GET /api/v1/analysis/{analysis_id}
```

Example response:

```json
{
  "analysis_id": "a83d1f",
  "status": "processing",
  "progress": 64,
  "frames_processed": 12840,
  "total_frames": 20000
}
```

Expected statuses:

```text
queued
processing
completed
failed
```

### Get analysis results

```http
GET /api/v1/analysis/{analysis_id}/results
```

Example:

```json
{
  "duration_seconds": 3600,
  "workers_detected": 18,
  "violations": {
    "no_helmet": 31,
    "no_safety_vest": 17,
    "no_boots": 4
  },
  "total_violation_events": 52,
  "violation_rate": 1.44
}
```

### Get violation timeline/events

```http
GET /api/v1/analysis/{analysis_id}/events
```

Example event:

```json
{
  "timestamp": 83.2,
  "type": "NO_HELMET",
  "confidence": 0.91,
  "track_id": 17,
  "duration": 4.8
}
```

### Download the PDF report

```http
GET /api/v1/analysis/{analysis_id}/report
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

Every significant violation should have at least one representative evidence frame saved:

```text
storage/
└── processed/
    └── <analysis_id>/
        └── events/
            ├── event_001.jpg
            ├── event_002.jpg
            └── event_003.jpg
```

Evidence should ideally contain the relevant worker and detection annotations. Evidence makes the generated report auditable instead of reducing it to a mysterious number produced by a neural network.

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

## Backend API Quickstart (API only, model.onnx usage via notebooks)

- Prerequisites: Python 3.9+ (preferred 3.10+). Ensure you have a working Python environment.
- Install dependencies:
  - From repo root: `python -m venv venv` (optional but recommended)
  - On macOS/Linux: `source venv/bin/activate`
  - Then install: `pip install -r backend/requirements.txt`
- Ensure the ONNX model is available at: `models/best.onnx` (path relative to repo root).
- Run the API server (backend only):
  - `uvicorn backend.app:app --reload --port 8000 --host 0.0.0.0`
- API endpoints (MVP):
  - POST /api/v1/analysis: submit a video (multipart) with site_name and camera_id. Returns analysis_id.
  - GET /api/v1/analysis/{analysis_id}: get analysis status and metadata.
  - GET /api/v1/analysis/{analysis_id}/results: get results payload.
  - GET /api/v1/analysis/{analysis_id}/events: get events payload.
  - GET /api/v1/analysis/{analysis_id}/report: download generated PDF report (if completed).
- Notes:
  - The MVP uses an in-memory store for analyses. For real deployments, migrate to a persistent DB.
  - The ONNX model can be exercised via the predict_frame function in backend/utils/predict.py; if the model cannot be loaded, a graceful fallback yields empty detections.
