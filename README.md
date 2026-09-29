# Facial Expression Recognition — End-to-End MLOps POC

This project takes a facial-expression CNN from a notebook to a deployable,
observable inference service. It uses CK+ for training, FastAPI for serving,
Docker and Amazon ECR for packaging, and Terraform to run a short-lived demo
on Amazon EKS behind an ALB and CloudFront.

## Current status

The end-to-end POC is implemented. It supports photos, a local webcam, and a
browser UI for webcam frames or uploaded images. The saved model can be
evaluated reproducibly on a subject-independent test split. The cloud
environment includes an immutable container, EKS deployment, HTTPS endpoint,
CloudWatch dashboard, alarms, and SNS email notifications. GitHub Actions
validates application and infrastructure changes on every pull request and
push to `main`.

```text
CK+ data -> training/evaluation -> versioned model
                                      |
Browser webcam/photo -> CloudFront -> ALB -> FastAPI on EKS -> prediction
                                      |
                             CloudWatch -> SNS email
```

This is a learning and portfolio POC, not a claim that facial-expression
classification is production-ready for sensitive or high-stakes use.

## Quick start

Prerequisites: Git, Python 3.9–3.12, and a compatible trained model. Python
3.11 is recommended because it matches the container. Model weights are not
stored in Git, so place the exported model at
`models/emotion_cnn_model.keras` before starting the API. Alternatively, use
the training command below if you have the CK+ archive.

```bash
git clone https://github.com/aadi308/facial-expression-recognition-mlops.git
cd facial-expression-recognition-mlops

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[api,dev]"

emotion-api
```

Open `http://127.0.0.1:8000`, then start the webcam or upload a photo. In a
second terminal, run the automated checks with:

```bash
source .venv/bin/activate
python -m pytest
```

If you need to create the model from `CK+.zip` instead of supplying an exported
model, install the training dependencies and run:

```bash
python -m pip install -e ".[api,training,dev]"
emotion-train --archive "CK+.zip"
emotion-evaluate --archive "CK+.zip"
```

For Docker and AWS deployment, first complete the local quick start, then
follow [`infra/terraform/README.md`](infra/terraform/README.md). Terraform
creates infrastructure; it does not train the model or build the image.

## Repository structure

```text
.
├── facial_emotion_recognition.ipynb  # Existing Colab training workflow
├── models/                           # Model metadata and validation report
├── Dockerfile                        # Non-root inference image
├── constraints-linux-py311.txt       # Exact deployed dependency set
├── .github/workflows/ci.yml          # Test and infrastructure CI checks
├── infra/                            # Terraform and scoped IAM policies
├── k8s/                              # Kubernetes base and POC overlay
├── src/emotion_recognition/
│   ├── config.py                     # Model input and label contract
│   ├── detection.py                  # OpenCV Haar face detection
│   ├── preprocessing.py              # Shared image transformation
│   ├── predictor.py                  # Model loading and prediction
│   ├── service.py                    # Detection + prediction orchestration
│   ├── api.py                        # FastAPI routes and application lifecycle
│   ├── api_cli.py                    # Local API server entry point
│   ├── training.py                   # Subject-independent local training
│   ├── evaluation.py                 # Saved-model validation and report
│   ├── monitoring.py                 # Prometheus inference metrics
│   ├── photo_cli.py                  # Full-photo inference entry point
│   ├── webcam_cli.py                 # Real-time webcam entry point
│   └── cli.py                        # Cropped-face inference entry point
├── tests/                            # Unit, API, and model integration tests
└── pyproject.toml                    # Package, dependencies, and CLI commands
```

## Model inference contract

The exported model must accept a batch with shape `(None, 100, 100, 1)` and
return eight softmax probabilities in this order:

1. Neutral
2. Anger
3. Contempt
4. Disgust
5. Fear
6. Happy
7. Sadness
8. Surprise

Inference loads an image as grayscale, resizes it to `100x100`, converts it to
`float32`, and normalizes each pixel from `0–255` to `0–1`.

## Train and export the model locally

Place `CK+.zip` in the repository root, then run:

```bash
emotion-train --archive "CK+.zip"
```

The command safely extracts only dataset PNGs into the ignored `data/`
directory, splits by subject, crops faces with the same detector used in live
inference, balances and augments only training data, evaluates unseen subjects,
and writes:

```text
models/emotion_cnn_model.keras
models/emotion_cnn_model.metadata.json
```

The model and dataset archives are ignored by Git. The small metadata JSON is
kept for experiment traceability.

## Validate the saved model

Evaluation is separate from training so a saved artifact can be checked again
without retraining it. With `CK+.zip` present and the dataset extracted, run:

```bash
emotion-evaluate --archive "CK+.zip"
```

This verifies the model input/output contract and probability output, checks
the artifact and dataset hashes, recreates the deterministic subject split,
and writes `models/model_validation.json`.

### Current validation result

The current artifact uses square-root class weighting selected through
three-fold, subject-grouped cross-validation. It was evaluated once on 92
images from 13 subjects that were not used for training, validation, or model
selection:

- Accuracy: `85.9%`
- Macro-F1: `61.6%`
- Balanced accuracy: `61.1%`
- Top-2 accuracy: `94.6%`
- Majority-class baseline: `63.0%`
- Subject-bootstrap 95% accuracy interval: `76.0%–94.6%`

The result is a substantial improvement over the original baseline, but the
classifier is still not production-grade. The test set has just 13 subjects;
fear recall is `0%` on two examples and sadness recall is `0%` on three. The
confidence interval also shows uncertainty from the small dataset. These
metrics are more honest than the original notebook result because no person's
images appear in more than one split and model selection used only development
subjects.

See `models/model_validation.json` for the confusion matrix, per-class
metrics, calibration result, hashes, environment versions, and exact split
counts. The cross-validation comparison is in `models/model_selection.json`,
and training metadata remains in `models/emotion_cnn_model.metadata.json`.

## Legacy Colab workflow

1. Open `facial_emotion_recognition.ipynb` in Google Colab.
2. Run the cells from top to bottom through the model-save cell.
3. Upload a CK+ zip containing these folders:

   - `neutral`
   - `anger`
   - `contempt`
   - `disgust`
   - `fear`
   - `happiness`
   - `sadness`
   - `surprise`

4. Run the download cell immediately after the model-save cell. It downloads
   `/content/emotion_cnn_model.keras` before the Colab runtime is deleted.
5. Place it locally at `models/emotion_cnn_model.keras`.

Model files are not committed to Git. The current model and metadata are also
stored as a versioned release in the private S3 artifact bucket; MLflow is a
later phase of this learning roadmap.

## Local setup

Python 3.9–3.12 is supported. Python 3.11 is used in the deployed container.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[api,dev]"
```

The training-only plotting and evaluation libraries can be installed with:

```bash
python -m pip install -e ".[training,dev]"
```

## Predict one image

The first inference interface expects an image containing one cropped face:

```bash
emotion-predict path/to/face.jpg
```

An explicit model path can be supplied when needed:

```bash
emotion-predict path/to/face.jpg --model path/to/model.keras
```

The equivalent module command is:

```bash
python -m emotion_recognition path/to/face.jpg
```

Successful output is JSON containing the predicted label, confidence, class
index, and all class probabilities.

## Detect and predict faces in a photo

Unlike `emotion-predict`, this command accepts a normal photo and first detects
faces within it:

```bash
emotion-photo path/to/photo.jpg --output output/annotated.jpg
```

It prints JSON for every detected face and optionally writes a copy of the
photo containing bounding boxes and predictions. To process only the largest
face, add `--largest-only`.

## Run real-time webcam inference

```bash
emotion-webcam
```

The webcam window displays face boxes, predicted expressions, confidence, and
an approximate FPS measurement. Press `q` or Escape to quit. If camera `0` is
not the correct device, try another index:

```bash
emotion-webcam --camera 1
```

On macOS, allow the terminal application to use the camera when prompted.

### Webcam troubleshooting on macOS

If the camera opens but no frames arrive:

1. Open **System Settings → Privacy & Security → Camera**.
2. Enable access for Terminal, iTerm, or the application running this command.
3. Close FaceTime, Zoom, browser video calls, and other camera applications.
4. Retry `emotion-webcam`, then try `emotion-webcam --camera 1` if needed.

The application waits up to five seconds for camera warm-up, tries the native
AVFoundation backend followed by OpenCV auto-detection, and tolerates brief
frame interruptions before exiting with an error.

## Run the prediction API

Start the local server from the repository root so the default model path can
be resolved:

```bash
emotion-api
```

If port 8000 is already in use, choose another local port with
`emotion-api --port 8001` and use that port in the URLs below.

The model is loaded once during application startup and then reused across
requests. Interactive API documentation is available at
`http://127.0.0.1:8000/docs`.

Check whether the process and model are ready:

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
curl http://127.0.0.1:8000/v1/model
curl http://127.0.0.1:8000/metrics
```

Upload a JPEG, PNG, WebP, or BMP photo for face detection and expression
classification:

```bash
curl -X POST \
  -F "file=@path/to/photo.jpg" \
  "http://127.0.0.1:8000/v1/predictions?largest_only=false"
```

The response contains a request ID, inference duration, face bounding boxes,
predicted classes, confidence values, and all class probabilities. An image
with no detected face is a successful request with `face_count: 0`; an invalid
file is a client error.

The default model, metadata, and 10 MiB upload limit can be configured without
changing code:

```bash
MODEL_PATH=path/to/model.keras \
MODEL_METADATA_PATH=path/to/model.metadata.json \
MAX_UPLOAD_BYTES=10485760 \
emotion-api --host 0.0.0.0 --port 8000
```

Binding the host process to `127.0.0.1` is appropriate for local development.
The container listens on all of its internal interfaces; later, the Kubernetes
Service and load balancer will control how it is exposed.

## Build and run the container

The container is the deployable unit stored in Amazon ECR and run on Amazon
EKS. Build it from the repository root:

```bash
docker build -t emotion-api:local .
```

The build context includes the trained model and its metadata, but excludes the
dataset, virtual environment, notebook, tests, Git history, and local output.
Run the image with a host port that is currently free:

```bash
docker run --rm --name emotion-api-local -p 8001:8000 emotion-api:local
```

The left side of `8001:8000` is the port on the laptop; the right side is the
port inside the container. Verify the container from another terminal:

```bash
curl http://127.0.0.1:8001/health/ready
curl http://127.0.0.1:8001/v1/model
curl -X POST \
  -F "file=@path/to/photo.jpg" \
  "http://127.0.0.1:8001/v1/predictions?largest_only=true"
```

The process runs as a non-root user, responds to termination signals, and has
a Docker health check backed by the API readiness endpoint. Python 3.11
dependencies are constrained to the versions validated in the deployed image.
The model is currently embedded in the image so one image identifies one exact
combination of application code and model weights. A later model-registry phase
will evaluate downloading a separately versioned model at startup.

On Apple Silicon, a normal local build uses `linux/arm64`. Cloud releases are
cross-built for `linux/amd64` so they work with standard EKS worker nodes. The
next immutable release is `emotion-api:v0.3.1`.

## Manage AWS infrastructure

Terraform is the source of truth for persistent AWS resources. The AWS CLI is
used for authentication, verification, imports, and troubleshooting—not for
creating normal project infrastructure.

The POC environment currently manages the private ECR repository, the private
versioned S3 model-artifact bucket, and their retention and security controls.
Terraform state is stored in a separate encrypted and versioned S3 backend:

```bash
AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc init \
  -backend-config=backend.hcl

AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc plan
```

See [`infra/terraform/README.md`](infra/terraform/README.md) for the current
infrastructure layout and workflow.

## Deploy the inference API to EKS

The POC Kubernetes overlay is prepared to deploy the immutable
`emotion-api:v0.3.1` image from ECR. Build and push that release before the
next deployment; CI validates the code but intentionally does not publish an
image or use AWS credentials. For a local-only demo, deploy the internal
Service and use port forwarding:

```bash
kubectl apply -k k8s/overlays/poc
kubectl rollout status deployment/emotion-api \
  --namespace emotion-api \
  --timeout 180s
kubectl get pods,service --namespace emotion-api
```

Forward the internal Service from one terminal:

```bash
kubectl port-forward \
  --namespace emotion-api \
  service/emotion-api \
  8001:80
```

Use another terminal to verify the model and submit a photo:

```bash
curl http://127.0.0.1:8001/health/ready
curl http://127.0.0.1:8001/v1/model
curl -X POST \
  -F "file=@path/to/photo.jpg" \
  "http://127.0.0.1:8001/v1/predictions?largest_only=true"
```

The Deployment runs as a non-root user with a read-only root filesystem,
resource requests and limits, restricted Pod Security, and Kubernetes startup,
readiness, and liveness probes. The model remains embedded in this baseline
image; the next model-delivery phase will use the existing EKS Pod Identity to
retrieve an explicitly versioned artifact from S3.

For the public web demo, Terraform installs the AWS Load Balancer
Controller, creates an ALB, and puts CloudFront in front of it for HTTPS. HTTPS
is required by browsers before they allow webcam access. Deploy and print the
generated URL with:

```bash
AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc-platform init \
  -backend-config=backend.hcl

AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc-platform plan \
  -out=tfplan

AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc-platform apply tfplan

terraform -chdir=infra/terraform/environments/poc-platform \
  output -raw public_url
```

Open that URL and select **Start camera** for live predictions, or select
**Upload image** to analyze a JPEG, PNG, WebP, or BMP photo. Both inputs use the
same FastAPI prediction endpoint, and uploaded images and video frames are not
stored. Destroy this short-lived public layer when the demo ends by following
the ordered teardown in [`infra/terraform/README.md`](infra/terraform/README.md).

## Monitoring and alerting

Terraform creates an operational dashboard for request volume, ALB/target
errors, inference-service latency, CloudFront traffic, and CloudFront 5xx rate.
It also creates four alarms:

- ALB-generated 5xx responses
- FastAPI target 5xx responses
- p95 target response time above three seconds
- CloudFront 5xx error rate above five percent

To enable email notifications, copy the example variable file before applying
the platform stack and confirm both AWS SNS subscription emails:

```bash
cp infra/terraform/environments/poc-platform/terraform.tfvars.example \
  infra/terraform/environments/poc-platform/terraform.tfvars
# Edit terraform.tfvars and set alert_email.

terraform -chdir=infra/terraform/environments/poc-platform \
  output -raw cloudwatch_dashboard_url
```

Two confirmations are expected because ALB metrics are regional in
`us-east-2`, while CloudFront metrics and its alarm live in `us-east-1`.
Monitoring resources are removed automatically when the platform stack is
destroyed.

The API also exposes Prometheus-compatible application and model-serving
metrics at `/metrics`:

- Prediction requests grouped by `success`, `no_face`, `rejected`, or `error`
- Inference latency histogram
- Total detected faces
- Predictions grouped by the eight fixed expression labels
- Prediction-confidence histogram

The labels have bounded cardinality and contain no filenames, request IDs,
images, or user identifiers. The endpoint provides live process metrics; a
Prometheus-compatible collector is still required for historical storage and
model-drift dashboards. In a production environment, `/metrics` should be
scraped internally instead of exposed through the public route.

## Continuous integration

`.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual
dispatches. It:

- installs the Python 3.11 application and runs the test suite;
- checks Terraform formatting and validates all three stacks without AWS
  credentials or a deployment;
- renders the Kubernetes POC overlay;
- validates every IAM policy as JSON; and
- fails if model binaries, datasets, Terraform state/plans/variables, or
  `.env` files are tracked.

The workflow has read-only repository permission and pins external actions to
immutable commits. CI performs validation only; it does not create AWS
resources or require AWS credentials.

## Run tests

```bash
python -m pytest
```

The suite tests image preprocessing, evaluation calculations, the model
contract, prediction output, API behavior, and loading a real Keras artifact. If
`models/emotion_cnn_model.keras` is present, it also validates the exported
model.


