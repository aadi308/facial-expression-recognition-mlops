import numpy as np
import pytest

from emotion_recognition.evaluation import (
    calculate_expected_calibration_error,
    calculate_top_k_accuracy,
    subject_bootstrap_intervals,
)


def test_top_k_accuracy_counts_labels_in_the_highest_probabilities():
    actual = np.asarray([0, 2])
    probabilities = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.5, 0.1, 0.4],
        ]
    )

    assert calculate_top_k_accuracy(actual, probabilities, k=1) == 0.5
    assert calculate_top_k_accuracy(actual, probabilities, k=2) == 1.0


def test_calibration_error_is_zero_for_certain_correct_predictions():
    actual = np.asarray([0, 1])
    probabilities = np.asarray([[1.0, 0.0], [0.0, 1.0]])

    assert calculate_expected_calibration_error(actual, probabilities) == 0.0


def test_subject_bootstrap_is_reproducible():
    actual = np.arange(8)
    predicted = actual.copy()
    subjects = [f"S{index:03d}" for index in range(8)]

    first = subject_bootstrap_intervals(
        actual, predicted, subjects, iterations=20, seed=42
    )
    second = subject_bootstrap_intervals(
        actual, predicted, subjects, iterations=20, seed=42
    )

    assert first == second
    assert first["accuracy_95_percent"] == pytest.approx([1.0, 1.0])
