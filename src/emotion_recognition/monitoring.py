"""Low-cardinality Prometheus metrics for the inference service."""

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Histogram


@dataclass(frozen=True)
class InferenceMetrics:
    registry: CollectorRegistry
    requests: Counter
    duration: Histogram
    faces: Counter
    predictions: Counter
    confidence: Histogram

    @classmethod
    def create(cls) -> "InferenceMetrics":
        registry = CollectorRegistry()
        return cls(
            registry=registry,
            requests=Counter(
                "emotion_prediction_requests_total",
                "Prediction requests grouped by outcome.",
                ("outcome",),
                registry=registry,
            ),
            duration=Histogram(
                "emotion_inference_duration_seconds",
                "Time spent detecting faces and running model inference.",
                buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10),
                registry=registry,
            ),
            faces=Counter(
                "emotion_faces_detected_total",
                "Faces detected in accepted prediction requests.",
                registry=registry,
            ),
            predictions=Counter(
                "emotion_predictions_total",
                "Model predictions grouped by visible-expression class.",
                ("expression",),
                registry=registry,
            ),
            confidence=Histogram(
                "emotion_prediction_confidence",
                "Maximum model probability for each face prediction.",
                buckets=(0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
                registry=registry,
            ),
        )
