"""Reproduce held-out model evaluation from a saved Keras artifact."""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from emotion_recognition.config import CLASS_NAMES
from emotion_recognition.training import (
    calculate_metrics,
    discover_records,
    load_images,
    sha256_file,
    split_records_by_subject,
    summarize_split,
)


def calculate_top_k_accuracy(
    actual: np.ndarray, probabilities: np.ndarray, k: int = 2
) -> float:
    if k <= 0 or k > probabilities.shape[1]:
        raise ValueError("k must be between 1 and the number of classes")
    top_k = np.argsort(probabilities, axis=1)[:, -k:]
    return float(np.mean([label in row for label, row in zip(actual, top_k)]))


def calculate_expected_calibration_error(
    actual: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    if bins <= 0:
        raise ValueError("bins must be positive")

    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = predicted == actual
    error = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            error += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return error


def subject_bootstrap_intervals(
    actual: np.ndarray,
    predicted: np.ndarray,
    subjects: Sequence[str],
    iterations: int,
    seed: int,
) -> Mapping[str, List[float]]:
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    if len(actual) != len(subjects):
        raise ValueError("one subject identifier is required for each prediction")

    indices_by_subject: Dict[str, List[int]] = defaultdict(list)
    for index, subject in enumerate(subjects):
        indices_by_subject[subject].append(index)

    unique_subjects = sorted(indices_by_subject)
    random_generator = np.random.default_rng(seed)
    accuracies: List[float] = []
    macro_f1_scores: List[float] = []
    for _ in range(iterations):
        sampled_subjects = random_generator.choice(
            unique_subjects, size=len(unique_subjects), replace=True
        )
        sampled_indices = np.concatenate(
            [
                np.asarray(indices_by_subject[str(subject)], dtype=np.int64)
                for subject in sampled_subjects
            ]
        )
        accuracy, macro_f1, _, _ = calculate_metrics(
            actual[sampled_indices], predicted[sampled_indices]
        )
        accuracies.append(accuracy)
        macro_f1_scores.append(macro_f1)

    return {
        "accuracy_95_percent": [
            float(np.percentile(accuracies, 2.5)),
            float(np.percentile(accuracies, 97.5)),
        ],
        "macro_f1_95_percent": [
            float(np.percentile(macro_f1_scores, 2.5)),
            float(np.percentile(macro_f1_scores, 97.5)),
        ],
    }


def evaluate_saved_model(
    model_path: Path,
    dataset_dir: Path,
    archive_path: Optional[Path] = None,
    seed: int = 42,
    batch_size: int = 32,
    bootstrap_iterations: int = 2000,
) -> Mapping[str, object]:
    import tensorflow as tf

    records = discover_records(dataset_dir)
    splits = split_records_by_subject(records, seed=seed)
    subject_sets = {
        name: {record.subject for record in split_records}
        for name, split_records in splits.items()
    }
    subject_overlap = {
        "train_validation": len(
            subject_sets["train"].intersection(subject_sets["validation"])
        ),
        "train_test": len(subject_sets["train"].intersection(subject_sets["test"])),
        "validation_test": len(
            subject_sets["validation"].intersection(subject_sets["test"])
        ),
    }
    if any(subject_overlap.values()):
        raise ValueError("Subject leakage detected between dataset splits")

    x_test, y_test = load_images(splits["test"])
    model = tf.keras.models.load_model(model_path, compile=False)
    probabilities = np.asarray(
        model.predict(x_test, batch_size=batch_size, verbose=0), dtype=np.float64
    )
    expected_shape = (len(y_test), len(CLASS_NAMES))
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"Model returned shape {probabilities.shape}; expected {expected_shape}"
        )
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("Model output is not a valid probability distribution")

    predictions = np.argmax(probabilities, axis=1)
    accuracy, macro_f1, confusion, per_class = calculate_metrics(
        y_test, predictions
    )
    confidence = np.max(probabilities, axis=1)
    correct = predictions == y_test
    class_counts = Counter(int(label) for label in y_test)
    majority_baseline = max(class_counts.values()) / len(y_test)

    artifact = {
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "input_shape": [dimension for dimension in model.input_shape],
        "output_shape": [dimension for dimension in model.output_shape],
        "parameters": int(model.count_params()),
    }
    if archive_path is not None and archive_path.is_file():
        artifact["dataset_archive_sha256"] = sha256_file(archive_path)

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": artifact,
        "dataset": {
            "images": len(records),
            "subjects": len({record.subject for record in records}),
            "splits": {
                name: summarize_split(split_records)
                for name, split_records in splits.items()
            },
            "subject_overlap": subject_overlap,
        },
        "evaluation": {
            "accuracy": accuracy,
            "majority_class_baseline_accuracy": majority_baseline,
            "macro_f1": macro_f1,
            "balanced_accuracy": float(
                np.mean([per_class[name]["recall"] for name in CLASS_NAMES])
            ),
            "top_2_accuracy": calculate_top_k_accuracy(y_test, probabilities, k=2),
            "expected_calibration_error_10_bins": (
                calculate_expected_calibration_error(y_test, probabilities, bins=10)
            ),
            "mean_confidence": float(confidence.mean()),
            "mean_confidence_correct": float(confidence[correct].mean()),
            "mean_confidence_incorrect": float(confidence[~correct].mean()),
            "subject_bootstrap": subject_bootstrap_intervals(
                y_test,
                predictions,
                [record.subject for record in splits["test"]],
                iterations=bootstrap_iterations,
                seed=seed,
            ),
            "confusion_matrix": confusion,
            "per_class": per_class,
        },
        "environment": {
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model on the subject-independent CK+ test split."
    )
    parser.add_argument(
        "--model", type=Path, default=Path("models/emotion_cnn_model.keras")
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/raw/ckplus")
    )
    parser.add_argument("--archive", type=Path, default=Path("CK+.zip"))
    parser.add_argument(
        "--output", type=Path, default=Path("models/model_validation.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_saved_model(
        model_path=args.model,
        dataset_dir=args.dataset_dir,
        archive_path=args.archive,
        seed=args.seed,
        batch_size=args.batch_size,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["evaluation"], indent=2))
    print(f"Saved validation report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
