"""Image loading and preprocessing shared by every inference interface."""

from pathlib import Path
from typing import Union

import cv2
import numpy as np

from emotion_recognition.config import IMAGE_SIZE


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Load an image as grayscale using the same behavior as the notebook."""

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Convert an image to a normalized `(1, 100, 100, 1)` model batch.

    Two-dimensional grayscale images and OpenCV-style BGR/BGRA arrays are
    accepted. Keeping this transformation in one function prevents training,
    CLI, and future API inference from drifting apart.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.size == 0:
        raise ValueError("image must not be empty")

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

    # INTER_LINEAR matches the default interpolation used by the notebook.
    resized = cv2.resize(grayscale, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    return normalized[np.newaxis, ..., np.newaxis]


def load_and_preprocess_image(image_path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk and prepare it for model prediction."""

    return preprocess_image(load_image(image_path))
