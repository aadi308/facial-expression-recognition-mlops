import cv2
import numpy as np
import pytest

from emotion_recognition.preprocessing import (
    load_and_preprocess_image,
    preprocess_image,
)


def test_preprocess_grayscale_image_matches_model_contract():
    image = np.full((40, 60), 255, dtype=np.uint8)

    batch = preprocess_image(image)

    assert batch.shape == (1, 100, 100, 1)
    assert batch.dtype == np.float32
    assert np.allclose(batch, 1.0)


def test_preprocess_accepts_bgr_image():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[..., 1] = 255

    batch = preprocess_image(image)

    assert batch.shape == (1, 100, 100, 1)
    assert 0.0 < float(batch.mean()) < 1.0


def test_load_and_preprocess_image(tmp_path):
    image_path = tmp_path / "face.png"
    assert cv2.imwrite(str(image_path), np.full((25, 25), 128, dtype=np.uint8))

    batch = load_and_preprocess_image(image_path)

    assert batch.shape == (1, 100, 100, 1)
    assert np.isclose(float(batch.mean()), 128 / 255)


@pytest.mark.parametrize(
    "image",
    [
        np.array([], dtype=np.uint8),
        np.zeros((10,), dtype=np.uint8),
        np.zeros((10, 10, 2), dtype=np.uint8),
    ],
)
def test_preprocess_rejects_invalid_images(image):
    with pytest.raises(ValueError):
        preprocess_image(image)
