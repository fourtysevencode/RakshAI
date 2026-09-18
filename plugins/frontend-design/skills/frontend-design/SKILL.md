Title: Frontend Design Skill
Author: OpenCode
Description: A pragmatic guide and implementation for building a Tailwind-based dark themed frontend with a yellow accent for uploading videos and retrieving PDF reports from the backend.

Overview
- The frontend is served by FastAPI from frontend_static/index.html at the root path ("/").
- It communicates with the backend via REST endpoints under /api/v1/analysis (POST to submit, GET to poll status) and /api/v1/analysis/{id}/report (PDF download).
- The UI uses a dark theme with a bold yellow accent to align with branding requirements.

Design goals
- A modern, accessible interface with a strong contrast: dark background with yellow highlights.
- Responsive layout that works on mobile and desktop.
- A minimal, robust integration with the backend; graceful handling of errors and loading states.
- No build step required; Tailwind is loaded via CDN for simplicity.

Implementation details
- Frontend assets live under frontend_static/. The root route serves index.html and the static assets are mounted under /frontend by FastAPI (unchanged).
- Tailwind CSS is loaded via CDN with a small Tailwind config to expose a brand color (amber/yellow).
- The UI collects: site_name, camera_id, frame_skip, and video file; submits to /api/v1/analysis.
- Results and PDF download are surfaced via /api/v1/analysis/{analysis_id} and /api/v1/analysis/{analysis_id}/report respectively.

Code sketch (high level)
- frontend_static/index.html: HTML + Tailwind-based styling; forms and JS for API calls.
- backend/app.py: API endpoints unchanged; documentation in code comments.

Testing approach
- Run the server via uvicorn and visit http://localhost:8000/.
- Use the form to upload a short video; observe progress updates and results in the UI.
- After completion, click the PDF report link to download the PDF.

Extensibility
- You can extend the UI with drag-and-drop for video upload, a video preview, or richer results visualization.
- If you later swap to a build step, you can swap the CDN Tailwind with a local build while preserving the API contract.
