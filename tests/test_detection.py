import numpy as np
import pytest

from emotion_recognition.detection import FaceBox, HaarFaceDetector, crop_face


def test_face_box_crops_expected_pixels():
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    face = FaceBox(x=2, y=3, width=4, height=2)

    crop = face.crop(image)

    assert crop.shape == (2, 4)
    assert np.array_equal(crop, image[3:5, 2:6])
    assert face.area == 8


def test_crop_face_adds_margin_and_clamps_to_image():
    image = np.zeros((20, 20), dtype=np.uint8)

    center_crop = crop_face(image, FaceBox(x=5, y=5, width=10, height=10))
    edge_crop = crop_face(image, FaceBox(x=0, y=0, width=10, height=10))

    assert center_crop.shape == (12, 12)
    assert edge_crop.shape == (11, 11)


def test_bundled_haar_cascade_loads_and_accepts_blank_image():
    detector = HaarFaceDetector()

    faces = detector.detect(np.zeros((200, 200, 3), dtype=np.uint8))

    assert faces == []


def test_detector_rejects_non_uint8_image():
    detector = HaarFaceDetector()

    with pytest.raises(ValueError, match="8-bit"):
        detector.detect(np.zeros((100, 100), dtype=np.float32))


def test_detector_validates_configuration():
    with pytest.raises(ValueError, match="scale_factor"):
        HaarFaceDetector(scale_factor=1.0)
