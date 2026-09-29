#!/usr/bin/env bash
set -euo pipefail

BACKEND_CMD="uvicorn backend.app:app --reload --port 8000 --host 0.0.0.0"

echo "Starting RakshAI backend. Frontend is served at the root URL (/)."
exec ${BACKEND_CMD}
