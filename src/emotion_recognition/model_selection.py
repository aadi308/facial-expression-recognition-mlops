"""Subject-group cross-validation for training-configuration selection."""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from emotion_recognition.config import CLASS_NAMES
from emotion_recognition.training import (
    build_model,
    calculate_frequency_class_weights,
    calculate_metrics,
    discover_records,
    extract_ckplus_images,
    load_images,
    split_records_by_subject,
    summarize_split,
)


def validate_weight_powers(weight_powers: Sequence[float]) -> List[float]:
    if not weight_powers:
        raise ValueError("At least one class-weight power is required")
    validated = [float(power) for power in weight_powers]
    if any(not 0.0 <= power <= 1.0 for power in validated):
        raise ValueError("Class-weight powers must be between 0 and 1")
    if len(set(validated)) != len(validated):
        raise ValueError("Class-weight powers must be unique")
    return validated


def cross_validate_class_weights(
    archive_path: Path,
    dataset_dir: Path,
    output_path: Path,
    weight_powers: Sequence[float],
    folds: int = 3,
    max_epochs: int = 20,
    batch_size: int = 32,
    seed: int = 42,
) -> Mapping[str, object]:
    """Select a class-weight power without evaluating the held-out test split."""

    import tensorflow as tf
    from sklearn.model_selection import StratifiedGroupKFold

    powers = validate_weight_powers(weight_powers)
    if folds < 2:
        raise ValueError("folds must be at least 2")

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    extract_ckplus_images(archive_path, dataset_dir)
    records = discover_records(dataset_dir)
    fixed_splits = split_records_by_subject(records, seed=seed)
    development_records = fixed_splits["train"] + fixed_splits["validation"]
    held_out_subjects = {record.subject for record in fixed_splits["test"]}
    if held_out_subjects.intersection(
        {record.subject for record in development_records}
    ):
        raise ValueError("Held-out test subjects leaked into model selection")

    images, labels = load_images(development_records)
    groups = np.asarray([record.subject for record in development_records])
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    fold_indices = list(splitter.split(images, labels, groups))

    candidates: List[Mapping[str, object]] = []
    for power in powers:
        fold_results: List[Mapping[str, object]] = []
        for fold_number, (train_indices, validation_indices) in enumerate(
            fold_indices, start=1
        ):
            fold_seed = seed + fold_number
            random.seed(fold_seed)
            np.random.seed(fold_seed)
            tf.random.set_seed(fold_seed)

            x_train, y_train = images[train_indices], labels[train_indices]
            x_validation = images[validation_indices]
            y_validation = labels[validation_indices]
            training_subjects = set(groups[train_indices])
            validation_subjects = set(groups[validation_indices])
            if training_subjects.intersection(validation_subjects):
                raise ValueError("Subject leakage detected inside a CV fold")

            model = build_model(seed=fold_seed)
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True,
                    verbose=0,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=2,
                    min_lr=1e-6,
                    verbose=0,
                ),
            ]
            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_validation, y_validation),
                epochs=max_epochs,
                batch_size=batch_size,
                class_weight=calculate_frequency_class_weights(
                    y_train, exponent=power
                ),
                callbacks=callbacks,
                verbose=0,
            )
            probabilities = model.predict(
                x_validation, batch_size=batch_size, verbose=0
            )
            predictions = np.argmax(probabilities, axis=1)
            accuracy, macro_f1, confusion, per_class = calculate_metrics(
                y_validation, predictions
            )
            fold_results.append(
                {
                    "fold": fold_number,
                    "epochs": len(history.history["loss"]),
                    "training_subjects": len(training_subjects),
                    "validation_subjects": len(validation_subjects),
                    "validation_images": len(validation_indices),
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "confusion_matrix": confusion,
                    "per_class": per_class,
                }
            )
            tf.keras.backend.clear_session()

        candidates.append(
            {
                "class_weight_power": power,
                "mean_accuracy": float(
                    np.mean([result["accuracy"] for result in fold_results])
                ),
                "std_accuracy": float(
                    np.std([result["accuracy"] for result in fold_results])
                ),
                "mean_macro_f1": float(
                    np.mean([result["macro_f1"] for result in fold_results])
                ),
                "std_macro_f1": float(
                    np.std([result["macro_f1"] for result in fold_results])
                ),
                "folds": fold_results,
            }
        )

    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["mean_macro_f1"], candidate["mean_accuracy"]
        ),
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "mean_macro_f1_then_mean_accuracy",
        "seed": seed,
        "fold_count": folds,
        "max_epochs": max_epochs,
        "development": {
            "images": len(development_records),
            "subjects": len(set(groups)),
        },
        "held_out_test": summarize_split(fixed_splits["test"]),
        "class_names": list(CLASS_NAMES),
        "candidates": candidates,
        "selected_class_weight_power": selected["class_weight_power"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved model-selection report: {output_path}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select class weighting with subject-group cross-validation."
    )
    parser.add_argument("--archive", type=Path, default=Path("CK+.zip"))
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/raw/ckplus")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/experiments/class_weight_selection.json"),
    )
    parser.add_argument(
        "--class-weight-powers",
        type=float,
        nargs="+",
        default=[0.5, 0.625, 0.75],
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cross_validate_class_weights(
        archive_path=args.archive,
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        weight_powers=args.class_weight_powers,
        folds=args.folds,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
