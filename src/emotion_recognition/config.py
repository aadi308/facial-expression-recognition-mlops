"""Shared inference contract for the trained CNN."""

from pathlib import Path
from typing import Tuple

IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
IMAGE_CHANNELS = 1
IMAGE_SIZE: Tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT)
MODEL_INPUT_SHAPE: Tuple[int, int, int] = (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS,
)

# The order must remain identical to the one-hot label order used during training.
CLASS_DIR_NAMES: Tuple[str, ...] = (
    "neutral",
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
)

CLASS_NAMES: Tuple[str, ...] = (
    "Neutral",
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Sadness",
    "Surprise",
)

DEFAULT_MODEL_PATH = Path("models/emotion_cnn_model.keras")
