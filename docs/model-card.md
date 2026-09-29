# Model card

## Model overview

The service uses a convolutional neural network to classify a detected face
into one of eight facial-expression classes: neutral, anger, contempt,
disgust, fear, happy, sadness, or surprise.

The model accepts a `100 x 100` grayscale image with pixel values scaled to
`0–1` and returns eight softmax probabilities. OpenCV face detection and
cropping run before classification.

## Intended use

The model is intended for demonstrating reproducible training, model
evaluation, API inference, container deployment, and monitoring. It can be
used with webcam frames or uploaded photos when users have consented to image
processing.

It must not be used to determine a person's internal emotional state or for
decisions involving employment, education, healthcare, law enforcement,
access control, or other high-impact use cases.

## Training data and split

Training uses the CK+ dataset prepared into 920 images from 123 subjects. The
split is performed by subject so images of one person cannot appear in more
than one partition.

| Split | Images | Subjects |
|---|---:|---:|
| Training | 738 | 98 |
| Validation | 90 | 12 |
| Test | 92 | 13 |

Only the training partition is augmented. Square-root inverse-frequency class
weights are used to reduce the effect of class imbalance. The weighting power
of `0.5` was selected with three-fold subject-grouped cross-validation on the
development data; the held-out test partition was not used for selection.

## Evaluation

The promoted model produced these results on the held-out test partition:

| Metric | Result |
|---|---:|
| Accuracy | 85.9% |
| Macro-F1 | 61.6% |
| Balanced accuracy | 61.1% |
| Top-2 accuracy | 94.6% |
| Expected calibration error, 10 bins | 8.5% |
| Subject-bootstrap accuracy interval, 95% | 76.0%–94.6% |

The majority-class baseline is 63.0%. Fear recall is 0% on two test examples,
and sadness recall is 0% on three test examples. The small supports make
per-class estimates unstable even where the reported score is high.

Full results are stored in:

- `models/emotion_cnn_model.metadata.json` — training configuration and model hash
- `models/model_selection.json` — cross-validation comparison
- `models/model_validation.json` — held-out metrics, calibration, and confusion matrix

## Limitations

- CK+ is small, controlled, and not representative of real-world camera,
  lighting, demographic, pose, occlusion, or cultural variation.
- A visible facial expression is not reliable evidence of a person's internal
  emotion, intent, or mental state.
- Haar-cascade face detection can fail on side profiles, small faces, poor
  lighting, and partial occlusion.
- Accuracy is dominated by the neutral class, so macro-F1 and per-class recall
  must be considered alongside overall accuracy.
- The current service does not detect distribution drift or automatically
  retrain the model.

## Reproduce validation

Place the CK+ archive at `CK+.zip`, install the training dependencies, and run:

```bash
python -m pip install -e ".[training,dev]"
emotion-evaluate --archive CK+.zip
```

The validation command checks the model contract, dataset hash, subject split,
artifact hash, and metrics. The promoted local artifact has SHA-256:

```text
1fb91ef29965c383d4e7ea907735ddcada7dfc33a5c6eb9be2b05d51f9499d9f
```

## Data handling

Uploaded images and webcam frames are processed in memory and are not stored
by the application. Metrics use fixed expression labels and do not include
images, filenames, request identifiers, or user identifiers.
