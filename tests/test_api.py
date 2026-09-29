import json
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from emotion_recognition.api import ApiSettings, create_app
from emotion_recognition.detection import FaceBox
from emotion_recognition.predictor import Prediction
from emotion_recognition.service import DetectedEmotion


class RecordingService:
    def __init__(self, detections=None):
        self.detections = detections or []
        self.calls = []

    def predict_faces(self, image, largest_only=False):
        self.calls.append(
            {"image_shape": image.shape, "largest_only": largest_only}
        )
        return self.detections


def make_settings(tmp_path: Path, max_upload_bytes: int = 1024 * 1024):
    return ApiSettings(
        model_path=tmp_path / "test-model.keras",
        metadata_path=tmp_path / "test-model.metadata.json",
        max_upload_bytes=max_upload_bytes,
    )


def encode_png() -> bytes:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def sample_detection() -> DetectedEmotion:
    return DetectedEmotion(
        face=FaceBox(x=4, y=5, width=16, height=18),
        prediction=Prediction(
            label="Happy",
            class_index=5,
            confidence=0.75,
            probabilities={"Happy": 0.75, "Neutral": 0.25},
        ),
    )


def test_webcam_ui_and_static_assets_are_served(tmp_path):
    app = create_app(
        settings=make_settings(tmp_path),
        service_loader=lambda _: RecordingService(),
    )

    with TestClient(app) as client:
        page = client.get("/")
        javascript = client.get("/static/app.js")
        stylesheet = client.get("/static/app.css")

    assert page.status_code == 200
    assert "Emotion Lens" in page.text
    assert 'id="camera"' in page.text
    assert 'id="image-input"' in page.text
    assert 'id="image-preview"' in page.text
    assert javascript.status_code == 200
    assert 'fetch("/v1/predictions?largest_only=true"' in javascript.text
    assert 'imageInput.addEventListener("change"' in javascript.text
    assert stylesheet.status_code == 200
    assert ".camera-stage" in stylesheet.text


def test_lifespan_loads_service_once_and_health_checks_are_ready(tmp_path):
    service = RecordingService()
    settings = make_settings(tmp_path)
    loaded_paths = []

    def loader(path):
        loaded_paths.append(path)
        return service

    app = create_app(settings=settings, service_loader=loader)

    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")

    assert loaded_paths == [settings.model_path]
    assert live.status_code == 200
    assert live.json() == {"status": "ok", "model_loaded": True}
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "model_loaded": True}
    assert app.state.ready is False
    assert app.state.service is None


def test_model_endpoint_exposes_artifact_metadata(tmp_path):
    settings = make_settings(tmp_path)
    settings.metadata_path.write_text(
        json.dumps(
            {
                "created_at_utc": "2026-08-21T21:33:37+00:00",
                "model": {
                    "sha256": "abc123",
                    "input_shape": [100, 100, 1],
                    "class_names": ["Neutral", "Happy"],
                },
                "evaluation": {
                    "test_accuracy": 0.66,
                    "test_macro_f1": 0.57,
                },
            }
        )
    )
    app = create_app(
        settings=settings, service_loader=lambda _: RecordingService()
    )

    with TestClient(app) as client:
        response = client.get("/v1/model")

    assert response.status_code == 200
    assert response.json() == {
        "model_path": str(settings.model_path),
        "model_sha256": "abc123",
        "created_at_utc": "2026-08-21T21:33:37+00:00",
        "input_shape": [100, 100, 1],
        "class_names": ["Neutral", "Happy"],
        "test_accuracy": 0.66,
        "test_macro_f1": 0.57,
    }


def test_prediction_endpoint_decodes_image_and_returns_detection(tmp_path):
    service = RecordingService([sample_detection()])
    app = create_app(
        settings=make_settings(tmp_path), service_loader=lambda _: service
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions?largest_only=true",
            files={"file": ("face.png", encode_png(), "image/png")},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["filename"] == "face.png"
    assert body["face_count"] == 1
    assert body["faces"][0]["face"] == {
        "x": 4,
        "y": 5,
        "width": 16,
        "height": 18,
    }
    assert body["faces"][0]["prediction"]["label"] == "Happy"
    assert body["inference_time_ms"] >= 0
    assert len(body["request_id"]) == 36
    assert service.calls == [
        {"image_shape": (24, 32, 3), "largest_only": True}
    ]


def test_metrics_endpoint_records_inference_without_request_details(tmp_path):
    service = RecordingService([sample_detection()])
    app = create_app(
        settings=make_settings(tmp_path), service_loader=lambda _: service
    )

    with TestClient(app) as client:
        client.post(
            "/v1/predictions",
            files={"file": ("face.png", encode_png(), "image/png")},
        )
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert (
        'emotion_prediction_requests_total{outcome="success"} 1.0'
        in response.text
    )
    assert 'emotion_predictions_total{expression="Happy"} 1.0' in response.text
    assert "emotion_faces_detected_total 1.0" in response.text
    assert "emotion_inference_duration_seconds_count 1.0" in response.text
    assert "emotion_prediction_confidence_count 1.0" in response.text
    assert "face.png" not in response.text


def test_prediction_endpoint_returns_empty_result_when_no_face_is_found(tmp_path):
    app = create_app(
        settings=make_settings(tmp_path),
        service_loader=lambda _: RecordingService(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions",
            files={"file": ("blank.png", encode_png(), "image/png")},
        )

    assert response.status_code == 200
    assert response.json()["face_count"] == 0
    assert response.json()["faces"] == []


def test_prediction_endpoint_rejects_unsupported_media_type(tmp_path):
    app = create_app(
        settings=make_settings(tmp_path),
        service_loader=lambda _: RecordingService(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions",
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )

    assert response.status_code == 415
    assert "Unsupported image type" in response.json()["detail"]


def test_prediction_endpoint_rejects_invalid_image_bytes(tmp_path):
    app = create_app(
        settings=make_settings(tmp_path),
        service_loader=lambda _: RecordingService(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions",
            files={"file": ("broken.png", b"not an image", "image/png")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a decodable image"


def test_prediction_endpoint_enforces_upload_size_limit(tmp_path):
    app = create_app(
        settings=make_settings(tmp_path, max_upload_bytes=8),
        service_loader=lambda _: RecordingService(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/predictions",
            files={"file": ("large.png", encode_png(), "image/png")},
        )

    assert response.status_code == 413
    assert "8 byte upload limit" in response.json()["detail"]
