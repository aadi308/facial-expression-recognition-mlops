from pathlib import Path

import numpy as np
import pytest

from emotion_recognition.config import CLASS_NAMES
from emotion_recognition.model_selection import validate_weight_powers
from emotion_recognition.training import (
    ImageRecord,
    balance_training_records,
    calculate_metrics,
    calculate_frequency_class_weights,
    split_records_by_subject,
)


def make_records():
    return [
        ImageRecord(
            path=Path(f"S{subject:03d}_{label}.png"),
            label=label,
            subject=f"S{subject:03d}",
        )
        for subject in range(40)
        for label in range(len(CLASS_NAMES))
    ]


def test_subject_split_has_no_subject_leakage_and_contains_every_class():
    splits = split_records_by_subject(make_records(), attempts=100)
    subject_sets = {
        name: {record.subject for record in records}
        for name, records in splits.items()
    }

    assert subject_sets["train"].isdisjoint(subject_sets["validation"])
    assert subject_sets["train"].isdisjoint(subject_sets["test"])
    assert subject_sets["validation"].isdisjoint(subject_sets["test"])
    for records in splits.values():
        assert {record.label for record in records} == set(range(len(CLASS_NAMES)))


def test_metrics_are_perfect_for_identical_predictions():
    labels = np.arange(len(CLASS_NAMES))

    accuracy, macro_f1, confusion, per_class = calculate_metrics(labels, labels)

    assert accuracy == 1.0
    assert macro_f1 == 1.0
    assert np.array_equal(confusion, np.eye(len(CLASS_NAMES), dtype=int))
    assert per_class["Happy"]["f1"] == 1.0


def test_training_balance_is_applied_after_subject_split():
    records = make_records()
    splits = split_records_by_subject(records, attempts=100)

    balanced = balance_training_records(splits["train"], 50, seed=42)

    counts = np.bincount(
        [record.label for record in balanced], minlength=len(CLASS_NAMES)
    )
    assert counts.tolist() == [50] * len(CLASS_NAMES)
    assert {record.subject for record in balanced}.issubset(
        {record.subject for record in splits["train"]}
    )


def test_subject_split_rejects_invalid_fractions():
    with pytest.raises(ValueError, match="sum to less than 1"):
        split_records_by_subject(
            make_records(), train_fraction=0.8, validation_fraction=0.2
        )


def test_frequency_class_weights_moderately_weight_rare_classes():
    labels = np.asarray(
        [0] * 64
        + [1] * 16
        + [2] * 4
        + [3] * 4
        + [4] * 4
        + [5] * 4
        + [6] * 4
        + [7] * 4
    )

    weights = calculate_frequency_class_weights(labels, exponent=0.5)

    assert weights[2] > weights[1] > weights[0]
    sample_weights = np.asarray([weights[int(label)] for label in labels])
    assert sample_weights.mean() == pytest.approx(1.0)


def test_class_weight_power_controls_minority_emphasis():
    labels = np.asarray(
        [0] * 64
        + [1] * 16
        + [2] * 4
        + [3] * 4
        + [4] * 4
        + [5] * 4
        + [6] * 4
        + [7] * 4
    )

    moderate = calculate_frequency_class_weights(labels, exponent=0.5)
    stronger = calculate_frequency_class_weights(labels, exponent=0.75)

    assert stronger[2] / stronger[0] > moderate[2] / moderate[0]
    with pytest.raises(ValueError, match="between 0 and 1"):
        calculate_frequency_class_weights(labels, exponent=1.1)


def test_training_cli_defaults_to_selected_class_weight_strategy():
    from emotion_recognition.training import build_parser

    args = build_parser().parse_args([])

    assert args.balance_strategy == "class-weight"
    assert args.class_weight_power == pytest.approx(0.5)


def test_model_selection_weight_powers_must_be_valid_and_unique():
    assert validate_weight_powers([0.5, 0.625, 0.75]) == [0.5, 0.625, 0.75]
    with pytest.raises(ValueError, match="unique"):
        validate_weight_powers([0.5, 0.5])
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_weight_powers([-0.1])
