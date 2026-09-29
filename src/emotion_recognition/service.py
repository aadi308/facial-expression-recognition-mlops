"""Application-level face detection and emotion prediction pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from emotion_recognition.detection import FaceBox, HaarFaceDetector, crop_face
from emotion_recognition.predictor import EmotionPredictor, Prediction


@dataclass(frozen=True)
class DetectedEmotion:
    """One detected face and its model prediction."""

    face: FaceBox
    prediction: Prediction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "face": self.face.to_dict(),
            "prediction": self.prediction.to_dict(),
        }


class EmotionRecognitionService:
    """Combine a face detector and model predictor for reusable inference."""

    def __init__(
        self,
        predictor: EmotionPredictor,
        detector: Optional[HaarFaceDetector] = None,
    ) -> None:
        self.predictor = predictor
        self.detector = detector or HaarFaceDetector()

    @classmethod
    def from_model_path(
        cls, model_path: Union[str, Path]
    ) -> "EmotionRecognitionService":
        return cls(predictor=EmotionPredictor.from_model_path(model_path))

    def predict_faces(
        self, image: np.ndarray, largest_only: bool = False
    ) -> List[DetectedEmotion]:
        faces = self.detector.detect(image)
        if largest_only:
            faces = faces[:1]

        return [
            DetectedEmotion(
                face=face,
                prediction=self.predictor.predict_array(crop_face(image, face)),
            )
            for face in faces
        ]
