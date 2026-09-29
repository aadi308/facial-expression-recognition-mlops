"""Model loading and facial emotion prediction."""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Mapping, Union

import numpy as np

from emotion_recognition.config import CLASS_NAMES, MODEL_INPUT_SHAPE
from emotion_recognition.preprocessing import (
    load_and_preprocess_image,
    preprocess_image,
)


@dataclass(frozen=True)
class Prediction:
    """Structured model output that is independent of HTTP or CLI concerns."""

    label: str
    class_index: int
    confidence: float
    probabilities: Mapping[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "class_index": self.class_index,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
        }


class EmotionPredictor:
    """Load a model once and reuse it for many predictions."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._prediction_lock = Lock()
        self._validate_model_contract()

    @classmethod
    def from_model_path(cls, model_path: Union[str, Path]) -> "EmotionPredictor":
        """Load a Keras model artifact without restoring training state."""

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Model file does not exist: {path}. "
                "Export emotion_cnn_model.keras from the training notebook first."
            )

        # TensorFlow is imported only when a real model must be loaded. This
        # keeps lightweight modules and unit tests fast to import.
        import tensorflow as tf

        model = tf.keras.models.load_model(path, compile=False)
        return cls(model)

    def predict_path(self, image_path: Union[str, Path]) -> Prediction:
        batch = load_and_preprocess_image(image_path)
        return self._predict_batch(batch)

    def predict_array(self, image: np.ndarray) -> Prediction:
        batch = preprocess_image(image)
        return self._predict_batch(batch)

    def _predict_batch(self, batch: np.ndarray) -> Prediction:
        # A single model instance is shared by API requests in one process.
        # Serializing predict calls avoids backend-specific thread-safety issues.
        with self._prediction_lock:
            raw_output = self._model.predict(batch, verbose=0)
        output = np.asarray(raw_output, dtype=np.float32)

        expected_shape = (1, len(CLASS_NAMES))
        if output.shape != expected_shape:
            raise ValueError(
                f"Model returned shape {output.shape}; expected {expected_shape}"
            )

        probabilities = output[0]
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Model returned non-finite probabilities")
        if np.any(probabilities < 0) or not np.isclose(
            float(probabilities.sum()), 1.0, atol=1e-3
        ):
            raise ValueError("Model output is not a valid probability distribution")

        class_index = int(np.argmax(probabilities))
        probability_map = {
            label: float(probabilities[index])
            for index, label in enumerate(CLASS_NAMES)
        }
        return Prediction(
            label=CLASS_NAMES[class_index],
            class_index=class_index,
            confidence=float(probabilities[class_index]),
            probabilities=probability_map,
        )

    def _validate_model_contract(self) -> None:
        input_shape = getattr(self._model, "input_shape", None)
        output_shape = getattr(self._model, "output_shape", None)

        if isinstance(input_shape, list) or input_shape is None:
            raise ValueError("Model must have exactly one input tensor")
        if isinstance(output_shape, list) or output_shape is None:
            raise ValueError("Model must have exactly one output tensor")

        actual_input_shape = tuple(input_shape[1:])
        if actual_input_shape != MODEL_INPUT_SHAPE:
            raise ValueError(
                f"Model input shape is {actual_input_shape}; "
                f"expected {MODEL_INPUT_SHAPE}"
            )

        output_classes = output_shape[-1]
        if output_classes != len(CLASS_NAMES):
            raise ValueError(
                f"Model has {output_classes} output classes; "
                f"expected {len(CLASS_NAMES)}"
            )
