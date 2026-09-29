# Resume project entry

## Recommended version

**Facial Expression Recognition MLOps POC** | Python, TensorFlow, FastAPI,
Docker, Terraform, AWS EKS

- Built a reproducible CNN training and evaluation workflow for CK+, using
  subject-disjoint splits to prevent the same person from appearing in train
  and test data; recorded artifact hashes, per-class metrics, calibration, and
  bootstrap confidence intervals.
- Packaged shared photo and webcam inference behind a tested FastAPI service
  and a non-root Docker image, with health probes, resource limits, immutable
  image releases in ECR, versioned model artifacts in S3, Prometheus-compatible
  inference metrics, and GitHub Actions validation.
- Provisioned a cost-conscious EKS demo with Terraform, ALB and CloudFront
  HTTPS access, plus CloudWatch traffic/error/latency dashboards and SNS email
  alerts; documented ordered teardown to avoid unnecessary AWS charges.

## Short version

**Facial Expression Recognition MLOps POC** | Python, TensorFlow, FastAPI,
Docker, Terraform, AWS

- Took a CNN from notebook training to a tested FastAPI service running on EKS,
  with reproducible subject-independent evaluation and versioned artifacts.
- Automated the AWS environment with Terraform and added HTTPS delivery,
  health checks, CloudWatch monitoring, and SNS alerting for a live webcam demo.

## Points to explain in an interview

- **Why subject-disjoint splitting matters:** adjacent CK+ frames from one
  person are very similar. Splitting by image can leak a person's appearance
  into the test set and inflate accuracy.
- **What the model result means:** held-out accuracy is `66.3%` and macro-F1 is
  `56.7%`. This is a valid baseline on a small, imbalanced dataset, not a
  production-grade emotion detector. The engineering workflow is the main
  portfolio result.
- **Why CloudFront is present:** browsers require a secure context for webcam
  access. CloudFront provides HTTPS while the short-lived POC uses an ALB
  origin.
- **Why infrastructure is split into Terraform states:** core resources and
  the application layer have different lifecycles. The public layer must be
  destroyed before EKS so Kubernetes can clean up its ALB.
- **What you would add next:** centralized structured logs, model/data-drift
  metrics, CI/CD, MLflow promotion, autoscaling, and load testing.

Do not describe the model as “perfect,” “highly accurate,” or suitable for
reading a person's true emotional state. The measured result and known limits
make the project more credible, not less.
