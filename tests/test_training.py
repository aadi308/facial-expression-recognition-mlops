from pathlib import Path

import numpy as np
import pytest

from emotion_recognition.config import CLASS_NAMES
from emotion_recognition.training import (
    ImageRecord,
    balance_training_records,
    calculate_metrics,
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
