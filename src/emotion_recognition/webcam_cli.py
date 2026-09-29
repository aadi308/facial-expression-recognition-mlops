"""Local real-time webcam inference interface."""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from emotion_recognition.config import DEFAULT_MODEL_PATH
from emotion_recognition.service import EmotionRecognitionService
from emotion_recognition.visualization import annotate_image


WINDOW_NAME = "Facial Expression Recognition"
STARTUP_ATTEMPTS = 50
STARTUP_RETRY_DELAY_SECONDS = 0.1
MAX_CONSECUTIVE_FRAME_FAILURES = 15


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run facial expression recognition from a local webcam."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))),
        help="Keras model path",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0)",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not mirror webcam frames horizontally",
    )
    return parser


def preferred_camera_backends() -> List[int]:
    """Prefer macOS AVFoundation, then fall back to OpenCV auto-detection."""

    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def read_frame_with_retry(
    capture,
    attempts: Optional[int] = None,
    retry_delay: Optional[float] = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> Optional[np.ndarray]:
    """Wait for a camera to warm up and return its first usable frame."""

    attempts = STARTUP_ATTEMPTS if attempts is None else attempts
    retry_delay = (
        STARTUP_RETRY_DELAY_SECONDS if retry_delay is None else retry_delay
    )
    for attempt in range(attempts):
        ok, frame = capture.read()
        if ok and frame is not None and frame.size > 0:
            return frame
        if attempt + 1 < attempts:
            sleeper(retry_delay)
    return None


def open_camera(
    camera_index: int,
    capture_factory=None,
) -> Tuple[object, np.ndarray]:
    """Open a camera using platform-appropriate backends and await a frame."""

    factory = capture_factory or cv2.VideoCapture
    opened_without_frames = False

    for backend in preferred_camera_backends():
        capture = factory(camera_index, backend)
        if not capture.isOpened():
            capture.release()
            continue

        first_frame = read_frame_with_retry(capture)
        if first_frame is not None:
            return capture, first_frame

        opened_without_frames = True
        capture.release()

    if opened_without_frames:
        raise RuntimeError(
            f"Camera {camera_index} opened but delivered no frames after "
            f"{STARTUP_ATTEMPTS} attempts. On macOS, enable camera access for "
            "your terminal in System Settings > Privacy & Security > Camera, "
            "close other camera apps, or try --camera 1."
        )
    raise RuntimeError(
        f"Could not open camera {camera_index}. Check macOS camera permissions, "
        "connect the camera, close other camera apps, or try --camera 1."
    )


def run_webcam(
    service: EmotionRecognitionService,
    camera_index: int = 0,
    mirror: bool = True,
) -> None:
    capture, first_frame = open_camera(camera_index)

    previous_time = time.perf_counter()
    smoothed_fps = 0.0
    consecutive_frame_failures = 0
    pending_frame: Optional[np.ndarray] = first_frame

    try:
        while True:
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
            else:
                ok, frame = capture.read()
                if not ok or frame is None or frame.size == 0:
                    consecutive_frame_failures += 1
                    if consecutive_frame_failures >= MAX_CONSECUTIVE_FRAME_FAILURES:
                        raise RuntimeError(
                            "The webcam repeatedly stopped returning frames. "
                            "Close other camera apps, reconnect the camera, or "
                            "try a different --camera index."
                        )
                    time.sleep(0.05)
                    continue
                consecutive_frame_failures = 0

            if mirror:
                frame = cv2.flip(frame, 1)

            detections = service.predict_faces(frame)
            annotated = annotate_image(frame, detections)

            current_time = time.perf_counter()
            elapsed = max(current_time - previous_time, 1e-9)
            current_fps = 1.0 / elapsed
            smoothed_fps = (
                current_fps
                if smoothed_fps == 0.0
                else 0.9 * smoothed_fps + 0.1 * current_fps
            )
            previous_time = current_time

            cv2.putText(
                annotated,
                f"FPS: {smoothed_fps:.1f} | q/esc to quit",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        service = EmotionRecognitionService.from_model_path(args.model)
        run_webcam(service, camera_index=args.camera, mirror=not args.no_mirror)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError, cv2.error) as exc:
        parser.exit(status=2, message=f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
