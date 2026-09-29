"""Lightweight OpenCV face detection for photos and webcam frames."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceBox:
    """Pixel coordinates for one detected face."""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y : self.y + self.height, self.x : self.x + self.width]

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


def crop_face(
    image: np.ndarray, face: FaceBox, margin_fraction: float = 0.1
) -> np.ndarray:
    """Crop a face with contextual margin while clamping to image bounds."""

    if margin_fraction < 0:
        raise ValueError("margin_fraction must not be negative")
    margin_x = round(face.width * margin_fraction)
    margin_y = round(face.height * margin_fraction)
    x_start = max(face.x - margin_x, 0)
    y_start = max(face.y - margin_y, 0)
    x_end = min(face.x + face.width + margin_x, image.shape[1])
    y_end = min(face.y + face.height + margin_y, image.shape[0])
    crop = image[y_start:y_end, x_start:x_end]
    if crop.size == 0:
        raise ValueError(f"Face box produced an empty crop: {face}")
    return crop


class HaarFaceDetector:
    """Detect frontal faces with OpenCV's bundled Haar cascade."""

    def __init__(
        self,
        cascade_path: Optional[Union[str, Path]] = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (40, 40),
    ) -> None:
        if scale_factor <= 1.0:
            raise ValueError("scale_factor must be greater than 1.0")
        if min_neighbors < 0:
            raise ValueError("min_neighbors must not be negative")

        if cascade_path is None:
            cascade_path = (
                Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            )

        self.cascade_path = Path(cascade_path)
        if not self.cascade_path.is_file():
            raise FileNotFoundError(f"Face cascade does not exist: {self.cascade_path}")

        self._classifier = cv2.CascadeClassifier(str(self.cascade_path))
        if self._classifier.empty():
            raise ValueError(f"Could not load face cascade: {self.cascade_path}")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, image: np.ndarray) -> List[FaceBox]:
        """Return detected faces ordered from largest to smallest."""

        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array")
        if image.size == 0:
            raise ValueError("image must not be empty")
        if image.dtype != np.uint8:
            raise ValueError("face detection expects an 8-bit image")

        if image.ndim == 2:
            grayscale = image
        elif image.ndim == 3 and image.shape[2] == 3:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(
                "image must be grayscale, BGR, or BGRA; "
                f"received shape {image.shape}"
            )

        equalized = cv2.equalizeHist(grayscale)
        raw_faces = self._classifier.detectMultiScale(
            equalized,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        boxes = [
            FaceBox(x=int(x), y=int(y), width=int(width), height=int(height))
            for x, y, width, height in raw_faces
        ]
        return sorted(boxes, key=lambda box: box.area, reverse=True)
