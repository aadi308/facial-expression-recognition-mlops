import numpy as np
import pytest

from emotion_recognition import webcam_cli


class FakeCapture:
    def __init__(self, frames, opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.frames:
            return False, None
        return self.frames.pop(0)

    def release(self):
        self.released = True


def test_read_frame_retries_transient_startup_failures():
    expected = np.zeros((10, 10, 3), dtype=np.uint8)
    capture = FakeCapture([(False, None), (False, None), (True, expected)])
    sleep_calls = []

    frame = webcam_cli.read_frame_with_retry(
        capture, attempts=3, retry_delay=0.01, sleeper=sleep_calls.append
    )

    assert frame is expected
    assert sleep_calls == [0.01, 0.01]


def test_open_camera_falls_back_to_another_backend(monkeypatch):
    expected = np.zeros((10, 10, 3), dtype=np.uint8)
    captures = [
        FakeCapture([], opened=False),
        FakeCapture([(True, expected)], opened=True),
    ]
    used_backends = []

    def factory(camera_index, backend):
        assert camera_index == 0
        used_backends.append(backend)
        return captures[len(used_backends) - 1]

    monkeypatch.setattr(webcam_cli, "preferred_camera_backends", lambda: [100, 200])

    capture, frame = webcam_cli.open_camera(0, capture_factory=factory)

    assert capture is captures[1]
    assert frame is expected
    assert captures[0].released
    assert used_backends == [100, 200]


def test_open_camera_reports_open_device_without_frames(monkeypatch):
    capture = FakeCapture([(False, None)] * 2, opened=True)
    monkeypatch.setattr(webcam_cli, "preferred_camera_backends", lambda: [100])
    monkeypatch.setattr(webcam_cli, "STARTUP_ATTEMPTS", 2)
    monkeypatch.setattr(webcam_cli, "STARTUP_RETRY_DELAY_SECONDS", 0.0)

    with pytest.raises(RuntimeError, match="opened but delivered no frames"):
        webcam_cli.open_camera(0, capture_factory=lambda camera, backend: capture)

    assert capture.released
