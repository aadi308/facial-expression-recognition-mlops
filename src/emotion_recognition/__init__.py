"""Facial emotion model inference package."""

from emotion_recognition.detection import FaceBox, HaarFaceDetector
from emotion_recognition.predictor import EmotionPredictor, Prediction
from emotion_recognition.service import DetectedEmotion, EmotionRecognitionService

__all__ = [
    "DetectedEmotion",
    "EmotionPredictor",
    "EmotionRecognitionService",
    "FaceBox",
    "HaarFaceDetector",
    "Prediction",
]
