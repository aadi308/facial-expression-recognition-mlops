"""FastAPI delivery layer for facial expression inference."""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool

from emotion_recognition.config import CLASS_NAMES, DEFAULT_MODEL_PATH
from emotion_recognition.monitoring import InferenceMetrics
from emotion_recognition.service import EmotionRecognitionService


ALLOWED_IMAGE_TYPES = {
    "image/bmp",
    "image/jpeg",
    "image/png",
    "image/webp",
}
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
WEB_DIRECTORY = Path(__file__).resolve().parent / "static"


@dataclass(frozen=True)
class ApiSettings:
    model_path: Path
    metadata_path: Path
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES

    @classmethod
    def from_environment(cls) -> "ApiSettings":
        model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        metadata_path = Path(
            os.getenv(
                "MODEL_METADATA_PATH",
                "models/emotion_cnn_model.metadata.json",
            )
        )
        raw_max_upload = os.getenv(
            "MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES)
        )
        try:
            max_upload_bytes = int(raw_max_upload)
        except ValueError as exc:
            raise ValueError("MAX_UPLOAD_BYTES must be an integer") from exc
        if max_upload_bytes <= 0:
            raise ValueError("MAX_UPLOAD_BYTES must be positive")
        return cls(
            model_path=model_path,
            metadata_path=metadata_path,
            max_upload_bytes=max_upload_bytes,
        )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class BoundingBoxResponse(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class EmotionPredictionResponse(BaseModel):
    label: str
    class_index: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    probabilities: Dict[str, float]


class DetectedFaceResponse(BaseModel):
    face: BoundingBoxResponse
    prediction: EmotionPredictionResponse


class ImagePredictionResponse(BaseModel):
    request_id: str
    filename: Optional[str]
    face_count: int = Field(ge=0)
    inference_time_ms: float = Field(ge=0)
    faces: List[DetectedFaceResponse]


class ModelInfoResponse(BaseModel):
    model_path: str
    model_sha256: Optional[str]
    created_at_utc: Optional[str]
    input_shape: List[int]
    class_names: List[str]
    test_accuracy: Optional[float]
    test_macro_f1: Optional[float]


ServiceLoader = Callable[[Path], EmotionRecognitionService]


def load_metadata(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Could not read model metadata: {path}") from exc


def decode_uploaded_image(payload: bytes) -> np.ndarray:
    if not payload:
        raise ValueError("Uploaded image is empty")
    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a decodable image")
    return image


def create_app(
    settings: Optional[ApiSettings] = None,
    service_loader: Optional[ServiceLoader] = None,
) -> FastAPI:
    resolved_settings = settings or ApiSettings.from_environment()
    resolved_loader = service_loader or EmotionRecognitionService.from_model_path
    metrics = InferenceMetrics.create()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.service = resolved_loader(resolved_settings.model_path)
        application.state.model_metadata = load_metadata(
            resolved_settings.metadata_path
        )
        application.state.ready = True
        try:
            yield
        finally:
            application.state.ready = False
            application.state.service = None

    application = FastAPI(
        title="Facial Expression Recognition API",
        version="0.3.1",
        description=(
            "Detect faces in an uploaded image and classify visible facial "
            "expressions with the configured Keras model."
        ),
        lifespan=lifespan,
    )
    application.mount(
        "/static",
        StaticFiles(directory=WEB_DIRECTORY),
        name="static",
    )

    @application.get("/", include_in_schema=False, response_class=FileResponse)
    def webcam_ui() -> FileResponse:
        return FileResponse(WEB_DIRECTORY / "index.html")

    @application.get("/health/live", response_model=HealthResponse)
    def liveness(request: Request) -> HealthResponse:
        model_loaded = getattr(request.app.state, "service", None) is not None
        return HealthResponse(status="ok", model_loaded=model_loaded)

    @application.get("/health/ready", response_model=HealthResponse)
    def readiness(request: Request) -> HealthResponse:
        ready = bool(getattr(request.app.state, "ready", False))
        if not ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready",
            )
        return HealthResponse(status="ready", model_loaded=True)

    @application.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(
            content=generate_latest(metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    @application.get("/v1/model", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        metadata: Mapping[str, Any] = request.app.state.model_metadata
        model_metadata = metadata.get("model", {})
        evaluation = metadata.get("evaluation", {})
        return ModelInfoResponse(
            model_path=str(resolved_settings.model_path),
            model_sha256=model_metadata.get("sha256"),
            created_at_utc=metadata.get("created_at_utc"),
            input_shape=list(model_metadata.get("input_shape", [100, 100, 1])),
            class_names=list(model_metadata.get("class_names", CLASS_NAMES)),
            test_accuracy=evaluation.get("test_accuracy"),
            test_macro_f1=evaluation.get("test_macro_f1"),
        )

    @application.post(
        "/v1/predictions",
        response_model=ImagePredictionResponse,
        status_code=status.HTTP_200_OK,
    )
    async def predict_image(
        request: Request,
        file: UploadFile = File(...),
        largest_only: bool = Query(False),
    ) -> ImagePredictionResponse:
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            metrics.requests.labels(outcome="rejected").inc()
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    "Unsupported image type. Use JPEG, PNG, WebP, or BMP."
                ),
            )

        payload = await file.read(resolved_settings.max_upload_bytes + 1)
        await file.close()
        if len(payload) > resolved_settings.max_upload_bytes:
            metrics.requests.labels(outcome="rejected").inc()
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(
                    f"Image exceeds the {resolved_settings.max_upload_bytes} "
                    "byte upload limit"
                ),
            )

        try:
            image = decode_uploaded_image(payload)
        except ValueError as exc:
            metrics.requests.labels(outcome="rejected").inc()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc

        service: EmotionRecognitionService = request.app.state.service
        started = time.perf_counter()
        try:
            detections = await run_in_threadpool(
                service.predict_faces, image, largest_only
            )
        except Exception:
            metrics.requests.labels(outcome="error").inc()
            raise

        inference_seconds = time.perf_counter() - started
        metrics.duration.observe(inference_seconds)
        metrics.faces.inc(len(detections))
        metrics.requests.labels(
            outcome="success" if detections else "no_face"
        ).inc()
        for detection in detections:
            metrics.predictions.labels(
                expression=detection.prediction.label
            ).inc()
            metrics.confidence.observe(detection.prediction.confidence)

        return ImagePredictionResponse(
            request_id=str(uuid.uuid4()),
            filename=file.filename,
            face_count=len(detections),
            inference_time_ms=inference_seconds * 1000,
            faces=[
                DetectedFaceResponse(**detection.to_dict())
                for detection in detections
            ],
        )

    return application


app = create_app()
