FROM python:3.11-slim-trixie@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

LABEL org.opencontainers.image.title="Facial Expression Recognition MLOps API" \
      org.opencontainers.image.description="FastAPI inference service for the CK+ CNN model"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODEL_PATH=/app/models/emotion_cnn_model.keras \
    MODEL_METADATA_PATH=/app/models/emotion_cnn_model.metadata.json \
    MAX_UPLOAD_BYTES=10485760

WORKDIR /app

# OpenCV's Python wheel dynamically loads these small runtime libraries.
RUN apt-get update \
    && apt-get upgrade --yes \
    && apt-get install --yes --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Installing dependencies before copying the model preserves Docker's layer
# cache when only the model artifact changes.
COPY pyproject.toml README.md constraints-linux-py311.txt ./
COPY src/ ./src/
RUN python -m pip install --upgrade "pip==26.2.1" \
    && python -m pip install --constraint constraints-linux-py311.txt ".[api]"

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

COPY --chown=10001:10001 models/emotion_cnn_model.keras ./models/emotion_cnn_model.keras
COPY --chown=10001:10001 models/emotion_cnn_model.metadata.json ./models/emotion_cnn_model.metadata.json

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)"]

STOPSIGNAL SIGTERM

CMD ["emotion-api", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
