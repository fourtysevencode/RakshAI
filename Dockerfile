FROM python:3.11-slim

# OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run containers as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
# CPU-only PyTorch first, so ultralytics doesn't pull the multi-GB CUDA build
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r backend/requirements.txt

COPY --chown=user backend backend
COPY --chown=user frontend_static frontend_static
COPY --chown=user models models
RUN mkdir -p storage/reports && chown -R user storage

USER user
ENV PORT=7860 YOLO_CONFIG_DIR=/tmp/Ultralytics
EXPOSE 7860
CMD uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}
