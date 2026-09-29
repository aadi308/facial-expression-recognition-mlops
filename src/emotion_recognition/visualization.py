"""Drawing helpers shared by photo and webcam interfaces."""

from typing import Iterable

import cv2
import numpy as np

from emotion_recognition.service import DetectedEmotion


def annotate_image(
    image: np.ndarray, detections: Iterable[DetectedEmotion]
) -> np.ndarray:
    """Return a copy with face boxes and predictions drawn on it."""

    annotated = image.copy()
    for detection in detections:
        face = detection.face
        prediction = detection.prediction
        top_left = (face.x, face.y)
        bottom_right = (face.x + face.width, face.y + face.height)
        cv2.rectangle(annotated, top_left, bottom_right, (0, 255, 0), 2)

        label = f"{prediction.label} {prediction.confidence:.0%}"
        text_y = max(face.y - 10, 20)
        cv2.putText(
            annotated,
            label,
            (face.x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated
