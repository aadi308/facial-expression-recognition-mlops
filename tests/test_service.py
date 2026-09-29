import numpy as np

from emotion_recognition.detection import FaceBox
from emotion_recognition.predictor import Prediction
from emotion_recognition.service import EmotionRecognitionService
from emotion_recognition.visualization import annotate_image


class FakeDetector:
    def detect(self, image):
        return [
            FaceBox(x=10, y=10, width=40, height=40),
            FaceBox(x=60, y=20, width=20, height=20),
        ]


class FakePredictor:
    def __init__(self):
        self.received_shapes = []

    def predict_array(self, image):
        self.received_shapes.append(image.shape)
        return Prediction(
            label="Happy",
            class_index=5,
            confidence=0.8,
            probabilities={"Happy": 0.8},
        )


def test_service_crops_and_predicts_each_detected_face():
    predictor = FakePredictor()
    service = EmotionRecognitionService(predictor=predictor, detector=FakeDetector())

    detections = service.predict_faces(np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(detections) == 2
    assert predictor.received_shapes == [(48, 48, 3), (24, 24, 3)]
    assert detections[0].prediction.label == "Happy"
    assert detections[0].to_dict()["face"]["width"] == 40


def test_service_can_limit_prediction_to_largest_face():
    predictor = FakePredictor()
    service = EmotionRecognitionService(predictor=predictor, detector=FakeDetector())

    detections = service.predict_faces(
        np.zeros((100, 100, 3), dtype=np.uint8), largest_only=True
    )

    assert len(detections) == 1
    assert predictor.received_shapes == [(48, 48, 3)]


def test_annotation_returns_modified_copy():
    predictor = FakePredictor()
    service = EmotionRecognitionService(predictor=predictor, detector=FakeDetector())
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = service.predict_faces(image, largest_only=True)

    annotated = annotate_image(image, detections)

    assert not np.shares_memory(image, annotated)
    assert np.array_equal(image, np.zeros_like(image))
    assert np.any(annotated != 0)
