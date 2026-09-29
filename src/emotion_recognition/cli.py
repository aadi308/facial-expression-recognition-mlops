"""Command-line entry point for local model inference."""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

from emotion_recognition.config import DEFAULT_MODEL_PATH
from emotion_recognition.predictor import EmotionPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict an emotion from one cropped face image."
    )
    parser.add_argument("image", type=Path, help="Path to the input face image")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))),
        help="Keras model path (default: MODEL_PATH or models/emotion_cnn_model.keras)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        predictor = EmotionPredictor.from_model_path(args.model)
        prediction = predictor.predict_path(args.image)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.exit(status=2, message=f"error: {exc}\n")

    print(json.dumps(prediction.to_dict(), indent=2, sort_keys=True))
    return 0
