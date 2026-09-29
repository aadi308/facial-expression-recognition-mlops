from pathlib import Path

import numpy as np
import pytest

from emotion_recognition.predictor import EmotionPredictor


def test_predictor_loads_keras_artifact_end_to_end(tmp_path):
    tf = pytest.importorskip("tensorflow")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(100, 100, 1)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(
                8,
                activation="softmax",
                kernel_initializer="zeros",
                bias_initializer="zeros",
            ),
        ]
    )
    model_path = tmp_path / "compatible_model.keras"
    model.save(model_path)

    predictor = EmotionPredictor.from_model_path(model_path)
    prediction = predictor.predict_array(np.zeros((100, 100), dtype=np.uint8))

    assert prediction.label == "Neutral"
    assert prediction.confidence == pytest.approx(0.125)


PROJECT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "emotion_cnn_model.keras"
)


@pytest.mark.skipif(
    not PROJECT_MODEL_PATH.exists(),
    reason="Exported notebook model is not available in models/ yet",
)
def test_exported_notebook_model_matches_inference_contract():
    predictor = EmotionPredictor.from_model_path(PROJECT_MODEL_PATH)

    prediction = predictor.predict_array(np.zeros((100, 100), dtype=np.uint8))

    assert len(prediction.probabilities) == 8
    assert 0.0 <= prediction.confidence <= 1.0
