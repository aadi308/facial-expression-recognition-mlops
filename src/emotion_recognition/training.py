"""Local, subject-independent training pipeline for the CK+ archive."""

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union
from zipfile import ZipFile

import cv2
import numpy as np

from emotion_recognition.config import (
    CLASS_DIR_NAMES,
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    IMAGE_SIZE,
    MODEL_INPUT_SHAPE,
)
from emotion_recognition.detection import HaarFaceDetector, crop_face


SUBJECT_PATTERN = re.compile(r"^(S\d+)_")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    subject: str


def extract_ckplus_images(archive_path: Path, dataset_dir: Path) -> int:
    """Extract only expected class PNGs, excluding archive executables/docs."""

    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive does not exist: {archive_path}")

    extracted = 0
    seen_targets: Set[Path] = set()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(archive_path) as archive:
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if info.is_dir() or member.suffix.lower() != ".png":
                continue
            if member.is_absolute() or ".." in member.parts or len(member.parts) < 2:
                raise ValueError(f"Unsafe archive member: {info.filename}")

            class_dir = member.parts[-2]
            if class_dir not in CLASS_DIR_NAMES:
                continue

            target = dataset_dir / class_dir / member.name
            if target in seen_targets:
                raise ValueError(f"Duplicate dataset image path: {target}")
            seen_targets.add(target)
            target.parent.mkdir(parents=True, exist_ok=True)

            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted += 1

    if extracted == 0:
        raise ValueError("No CK+ class PNGs were found in the archive")
    return extracted


def discover_records(dataset_dir: Path) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for label, class_dir in enumerate(CLASS_DIR_NAMES):
        for path in sorted((dataset_dir / class_dir).glob("*.png")):
            match = SUBJECT_PATTERN.match(path.stem)
            if match is None:
                raise ValueError(f"Cannot determine subject ID from filename: {path.name}")
            records.append(
                ImageRecord(path=path, label=label, subject=match.group(1))
            )

    if not records:
        raise ValueError(f"No dataset images found under {dataset_dir}")
    return records


def split_records_by_subject(
    records: Sequence[ImageRecord],
    seed: int = 42,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    attempts: int = 3000,
) -> Dict[str, List[ImageRecord]]:
    """Choose disjoint subject splits while keeping every class represented."""

    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")

    subjects = sorted({record.subject for record in records})
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required")

    number_of_classes = len(CLASS_NAMES)
    total_class_counts = np.bincount(
        [record.label for record in records], minlength=number_of_classes
    ).astype(np.float64)
    if np.any(total_class_counts == 0):
        raise ValueError("Every configured class must contain at least one image")

    records_by_subject: Dict[str, List[ImageRecord]] = {}
    for subject in subjects:
        records_by_subject[subject] = [
            record for record in records if record.subject == subject
        ]

    train_subject_count = max(1, round(len(subjects) * train_fraction))
    validation_subject_count = max(1, round(len(subjects) * validation_fraction))
    if train_subject_count + validation_subject_count >= len(subjects):
        raise ValueError("Not enough subjects for three non-empty splits")

    split_fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": 1.0 - train_fraction - validation_fraction,
    }
    random_generator = random.Random(seed)
    best_subjects: Optional[Dict[str, Set[str]]] = None
    best_score = float("inf")

    for _ in range(attempts):
        shuffled = subjects.copy()
        random_generator.shuffle(shuffled)
        subject_splits = {
            "train": set(shuffled[:train_subject_count]),
            "validation": set(
                shuffled[
                    train_subject_count : train_subject_count
                    + validation_subject_count
                ]
            ),
            "test": set(shuffled[train_subject_count + validation_subject_count :]),
        }

        score = 0.0
        valid = True
        for split_name, split_subjects in subject_splits.items():
            labels = [
                record.label
                for subject in split_subjects
                for record in records_by_subject[subject]
            ]
            counts = np.bincount(labels, minlength=number_of_classes)
            if np.any(counts == 0):
                valid = False
                break
            actual_fractions = counts / total_class_counts
            score += float(
                np.square(actual_fractions - split_fractions[split_name]).sum()
            )

        if valid and score < best_score:
            best_score = score
            best_subjects = subject_splits

    if best_subjects is None:
        raise ValueError(
            "Could not find subject-independent splits containing every class"
        )

    return {
        split_name: sorted(
            [record for record in records if record.subject in split_subjects],
            key=lambda record: str(record.path),
        )
        for split_name, split_subjects in best_subjects.items()
    }


def load_images(records: Sequence[ImageRecord]) -> Tuple[np.ndarray, np.ndarray]:
    images: List[np.ndarray] = []
    labels: List[int] = []
    detector = HaarFaceDetector(min_size=(80, 80))
    for record in records:
        image = cv2.imread(str(record.path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"OpenCV could not decode image: {record.path}")
        detected_faces = detector.detect(image)
        if not detected_faces:
            raise ValueError(f"No face detected in training image: {record.path}")
        face_image = crop_face(image, detected_faces[0])
        resized = cv2.resize(face_image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        images.append(resized.astype(np.float32) / 255.0)
        labels.append(record.label)

    image_array = np.asarray(images, dtype=np.float32)[..., np.newaxis]
    label_array = np.asarray(labels, dtype=np.int64)
    return image_array, label_array


def balance_training_records(
    records: Sequence[ImageRecord], samples_per_class: int, seed: int
) -> List[ImageRecord]:
    """Downsample or repeat training records to create equal class counts.

    Repetition is safe here because subject splitting has already happened and
    the model applies random augmentation each time a repeated image is seen.
    """

    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")

    random_generator = random.Random(seed)
    balanced: List[ImageRecord] = []
    for label in range(len(CLASS_NAMES)):
        class_records = [record for record in records if record.label == label]
        if not class_records:
            raise ValueError(f"Training split has no images for class {label}")

        if len(class_records) >= samples_per_class:
            selected = random_generator.sample(class_records, samples_per_class)
        else:
            selected = class_records.copy()
            while len(selected) < samples_per_class:
                selected.append(random_generator.choice(class_records))
        balanced.extend(selected)

    random_generator.shuffle(balanced)
    return balanced


def build_model(seed: int = 42):
    import tensorflow as tf

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(12 / 360, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomContrast(0.1, seed=seed),
        ],
        name="training_augmentation",
    )

    inputs = tf.keras.layers.Input(shape=MODEL_INPUT_SHAPE)
    x = augmentation(inputs)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="emotion_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def calculate_class_weights(labels: np.ndarray) -> Dict[int, float]:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    if np.any(counts == 0):
        raise ValueError("Training split must contain every class")
    return {
        index: float(len(labels) / (len(CLASS_NAMES) * count))
        for index, count in enumerate(counts)
    }


def calculate_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> Tuple[float, float, List[List[int]], Dict[str, Mapping[str, float]]]:
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for actual_label, predicted_label in zip(actual, predicted):
        confusion[int(actual_label), int(predicted_label)] += 1

    per_class: Dict[str, Mapping[str, float]] = {}
    f1_scores: List[float] = []
    for index, class_name in enumerate(CLASS_NAMES):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_scores.append(f1)
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[index, :].sum()),
        }

    accuracy = float(np.mean(actual == predicted))
    macro_f1 = float(np.mean(f1_scores))
    return accuracy, macro_f1, confusion.tolist(), per_class


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_split(records: Sequence[ImageRecord]) -> Mapping[str, object]:
    counts = Counter(record.label for record in records)
    return {
        "images": len(records),
        "subjects": len({record.subject for record in records}),
        "class_counts": {
            CLASS_NAMES[index]: counts[index] for index in range(len(CLASS_NAMES))
        },
    }


def train(
    archive_path: Path,
    dataset_dir: Path,
    model_output: Path,
    metadata_output: Path,
    max_epochs: int,
    batch_size: int,
    seed: int,
    samples_per_class: int,
) -> Mapping[str, object]:
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    extracted_count = extract_ckplus_images(archive_path, dataset_dir)
    records = discover_records(dataset_dir)
    if extracted_count != len(records):
        raise ValueError(
            f"Extracted {extracted_count} images but discovered {len(records)}"
        )

    splits = split_records_by_subject(records, seed=seed)
    split_summaries = {
        split_name: summarize_split(split_records)
        for split_name, split_records in splits.items()
    }
    print(json.dumps({"splits": split_summaries}, indent=2))

    balanced_training_records = balance_training_records(
        splits["train"], samples_per_class=samples_per_class, seed=seed
    )
    x_train, y_train = load_images(balanced_training_records)
    x_validation, y_validation = load_images(splits["validation"])
    x_test, y_test = load_images(splits["test"])
    validation_class_weights = calculate_class_weights(y_validation)
    validation_sample_weights = np.asarray(
        [validation_class_weights[int(label)] for label in y_validation],
        dtype=np.float32,
    )

    model = build_model(seed=seed)
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation, validation_sample_weights),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    probabilities = model.predict(x_test, batch_size=batch_size, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy, macro_f1, confusion, per_class = calculate_metrics(
        y_test, predictions
    )

    model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_output)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "archive": str(archive_path),
            "archive_sha256": sha256_file(archive_path),
            "images": len(records),
            "subjects": len({record.subject for record in records}),
        },
        "model": {
            "path": str(model_output),
            "sha256": sha256_file(model_output),
            "input_shape": list(MODEL_INPUT_SHAPE),
            "class_names": list(CLASS_NAMES),
            "parameters": int(model.count_params()),
        },
        "training": {
            "seed": seed,
            "batch_size": batch_size,
            "balanced_samples_per_class": samples_per_class,
            "requested_epochs": max_epochs,
            "completed_epochs": len(history.history["loss"]),
            "splits": split_summaries,
            "preprocessing": {
                "face_detector": "haarcascade_frontalface_default.xml",
                "face_margin_fraction": 0.1,
                "grayscale": True,
                "resize": list(IMAGE_SIZE),
                "pixel_scaling": "divide_by_255",
            },
        },
        "evaluation": {
            "test_accuracy": accuracy,
            "test_macro_f1": macro_f1,
            "confusion_matrix": confusion,
            "per_class": per_class,
        },
        "environment": {
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["evaluation"], indent=2))
    print(f"Saved model: {model_output}")
    print(f"Saved metadata: {metadata_output}")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a subject-independent CK+ emotion classifier."
    )
    parser.add_argument("--archive", type=Path, default=Path("CK+.zip"))
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/raw/ckplus")
    )
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("models/emotion_cnn_model.metadata.json"),
    )
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    train(
        archive_path=args.archive,
        dataset_dir=args.dataset_dir,
        model_output=args.model_output,
        metadata_output=args.metadata_output,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        samples_per_class=args.samples_per_class,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
