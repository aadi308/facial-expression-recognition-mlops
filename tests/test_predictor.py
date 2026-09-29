import numpy as np
import pytest

from emotion_recognition.config import CLASS_NAMES
from emotion_recognition.predictor import EmotionPredictor


class FakeModel:
    input_shape = (None, 100, 100, 1)
    output_shape = (None, 8)

    def __init__(self, probabilities=None):
        if probabilities is None:
            probabilities = [0.01, 0.02, 0.03, 0.04, 0.05, 0.8, 0.02, 0.03]
        self.probabilities = np.asarray([probabilities], dtype=np.float32)
        self.last_batch = None

    def predict(self, batch, verbose=0):
        self.last_batch = batch
        return self.probabilities


def test_predictor_returns_structured_prediction():
    model = FakeModel()
    predictor = EmotionPredictor(model)

    prediction = predictor.predict_array(np.zeros((50, 50), dtype=np.uint8))

    assert prediction.label == "Happy"
    assert prediction.class_index == 5
    assert prediction.confidence == pytest.approx(0.8)
    assert tuple(prediction.probabilities) == CLASS_NAMES
    assert model.last_batch.shape == (1, 100, 100, 1)


def test_predictor_rejects_wrong_model_input_shape():
    model = FakeModel()
    model.input_shape = (None, 48, 48, 1)

    with pytest.raises(ValueError, match="input shape"):
        EmotionPredictor(model)


def test_predictor_rejects_wrong_number_of_classes():
    model = FakeModel()
    model.output_shape = (None, 7)

    with pytest.raises(ValueError, match="output classes"):
        EmotionPredictor(model)


def test_predictor_rejects_non_probability_output():
    model = FakeModel([1.0] * 8)
    predictor = EmotionPredictor(model)

    with pytest.raises(ValueError, match="probability distribution"):
        predictor.predict_array(np.zeros((100, 100), dtype=np.uint8))
