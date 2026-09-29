"""Detect faces and predict facial expressions in a photo."""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import cv2

from emotion_recognition.config import DEFAULT_MODEL_PATH
from emotion_recognition.service import EmotionRecognitionService
from emotion_recognition.visualization import annotate_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect faces and predict expressions in one photo."
    )
    parser.add_argument("image", type=Path, help="Path to the input photo")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))),
        help="Keras model path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for an annotated output image",
    )
    parser.add_argument(
        "--largest-only",
        action="store_true",
        help="Predict only the largest detected face",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.exit(status=2, message=f"error: could not read image: {args.image}\n")

    try:
        service = EmotionRecognitionService.from_model_path(args.model)
        detections = service.predict_faces(image, largest_only=args.largest_only)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.exit(status=2, message=f"error: {exc}\n")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), annotate_image(image, detections)):
            parser.exit(
                status=2,
                message=f"error: could not write output image: {args.output}\n",
            )

    payload = {
        "image": str(args.image),
        "face_count": len(detections),
        "faces": [detection.to_dict() for detection in detections],
    }
    if args.output is not None:
        payload["output"] = str(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
