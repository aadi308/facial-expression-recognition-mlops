# System architecture

## Request flow

```text
Browser webcam or image upload
              |
              v
CloudFront HTTPS -> ALB -> Kubernetes Service -> FastAPI pod
                                                   |
                                                   v
                                      face detection and CNN inference
                                                   |
                                                   v
                                          JSON prediction response
```

For local development, clients connect directly to FastAPI. In the AWS demo,
CloudFront provides HTTPS, the Application Load Balancer routes traffic into
Amazon EKS, and Kubernetes sends requests to the inference pod.

## Application components

- `detection.py` locates faces with OpenCV and applies a configurable margin.
- `preprocessing.py` converts each face to the model's grayscale `100 x 100`
  input contract.
- `predictor.py` loads the Keras model once and returns class probabilities.
- `service.py` coordinates detection and prediction for photos and frames.
- `api.py` exposes the UI, prediction endpoint, health probes, model metadata,
  and Prometheus metrics.
- `training.py`, `model_selection.py`, and `evaluation.py` keep training,
  model selection, and final validation separate.

The command-line photo and webcam clients use the same detection,
preprocessing, and prediction modules as the API. This avoids different model
behavior between local testing and deployment.

## Artifact flow

```text
CK+ archive
    |
    v
subject-disjoint training -> candidate model
                                  |
                       grouped cross-validation
                                  |
                                  v
                         selected model artifact
                                  |
                         held-out validation
                                  |
                 metadata + validation + SHA-256
                                  |
                        Docker image / S3 artifact
```

Model binaries and datasets are not committed to Git. Metadata, selection
evidence, validation results, and artifact hashes are committed so a release
can be traced to its training and evaluation results. The deployment image
contains one exact model artifact and is published to Amazon ECR with an
immutable version tag.

## Infrastructure

Terraform is divided by lifecycle:

- `bootstrap` creates the remote Terraform state bucket.
- `poc` creates persistent regional resources including networking, ECR, the
  model-artifact S3 bucket, IAM roles, and EKS.
- `poc-platform` installs cluster-facing resources, deploys the application,
  and creates ALB, CloudFront, CloudWatch, and SNS resources.

The platform layer is destroyed before the EKS layer so the AWS Load Balancer
Controller can remove its managed load balancer resources cleanly.

## Runtime controls

- The container runs as a non-root user with a read-only root filesystem.
- Kubernetes defines CPU and memory requests and limits.
- Startup, readiness, and liveness probes use dedicated API endpoints.
- CloudFront supplies HTTPS for browser camera access.
- The S3 artifact bucket blocks public access and uses versioning and
  encryption.
- IAM policies scope application access to the required AWS resources.

## Observability

FastAPI exposes Prometheus-compatible request, latency, face-count,
prediction-label, and confidence metrics. CloudWatch monitors CloudFront and
ALB traffic, errors, and latency. SNS delivers alarm notifications after the
email subscription is confirmed.

Images and user identifiers are deliberately excluded from metrics. A metrics
collector is required for historical Prometheus dashboards; the application
endpoint alone only exposes the current process state.

## Deployment references

- Local setup and application commands: [`../README.md`](../README.md)
- Terraform deployment and teardown: [`../infra/terraform/README.md`](../infra/terraform/README.md)
- Kubernetes manifests: [`../k8s/`](../k8s/)
- CI validation: [`../.github/workflows/ci.yml`](../.github/workflows/ci.yml)
