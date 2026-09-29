# AWS deployment with Terraform

Terraform is the source of truth for persistent AWS resources in this project.
The AWS CLI is used only for authentication, read-only verification, imports,
and troubleshooting.

Terraform does not train the model or build the application. It creates the AWS
infrastructure that stores and runs the Dockerized inference API:

```text
TensorFlow model + FastAPI application
                 -> Docker image
                 -> Amazon ECR
                 -> Amazon EKS
                 -> ALB -> CloudFront HTTPS

Terraform        -> S3, ECR, VPC, IAM, EKS, ALB, and CloudFront
Monitoring       -> CloudWatch dashboard/alarms -> SNS email
```

## Prerequisites

- AWS account and a non-root IAM user or assumable role.
- AWS CLI profile named `emotio-mlops-user` in region `us-east-2`.
- Terraform 1.10 or newer. This project currently uses Terraform 1.16.3.
- Docker Desktop for building and testing the API image.
- The trained model at `models/emotion_cnn_model.keras`.
- `kubectl` is used for deployment verification. Terraform installs the Helm
  chart through its Helm provider, so a local Helm CLI is optional.

Verify the local tools and AWS identity:

```bash
aws sts get-caller-identity --profile emotio-mlops-user
terraform version
docker version
```

The checked-in backend files, IAM policies, bucket names, and default ECR image
refer to the portfolio owner's AWS account. If you fork this repository, replace
account ID `619759452242`, choose globally unique S3 bucket names, and use your
own AWS CLI profile before applying. AWS account IDs are identifiers, not
credentials; never commit access keys or secret keys.

## Layout

- `modules/`: reusable infrastructure components.
- `bootstrap/`: encrypted and versioned S3 storage for Terraform state.
- `environments/poc/`: the low-cost learning environment in `us-east-2`.
- `environments/poc-platform/`: the Kubernetes application, AWS Load Balancer
  Controller, public ALB, CloudFront HTTPS endpoint, CloudWatch monitoring,
  and SNS alert subscriptions.

The existing `emotion-api` ECR repository was initially created with the AWS
CLI. Declarative `import` blocks in the POC environment adopt it into Terraform
without deleting the repository or its images.

## Step 1: bootstrap remote Terraform state

The bootstrap stack uses local state only while it creates the S3 backend. Its
state is migrated into that bucket immediately afterward.

```bash
terraform -chdir=infra/terraform/bootstrap init
terraform -chdir=infra/terraform/bootstrap fmt -check
terraform -chdir=infra/terraform/bootstrap validate
terraform -chdir=infra/terraform/bootstrap plan -out=tfplan
terraform -chdir=infra/terraform/bootstrap apply tfplan

AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/bootstrap init \
  -migrate-state \
  -backend-config=backend.hcl
```

The executing identity needs the permissions in
`infra/iam/terraform-s3-policy.json` before running `apply`.

The completed bootstrap state is stored at
`s3://emotion-mlops-tfstate-619759452242-us-east-2/bootstrap/terraform.tfstate`.

## Step 2: initialize the POC environment

```bash
AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc init \
  -migrate-state \
  -backend-config=backend.hcl

AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc plan -out=tfplan

AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc apply tfplan
```

The POC state is stored separately at
`s3://emotion-mlops-tfstate-619759452242-us-east-2/environments/poc/terraform.tfstate`.

## Step 3: publish a model artifact

Terraform creates and secures the model-artifact bucket. Model files are
versioned release artifacts, so they are published by an artifact command now
and by CI or MLflow later; they are not stored in Terraform state.

The current release layout is:

```text
s3://emotion-mlops-artifacts-619759452242-us-east-2/
└── models/emotion-cnn/v1/
    ├── model.keras
    └── metadata.json
```

Publish the model and its metadata with full-object SHA-256 checksums:

```bash
aws s3api put-object \
  --bucket emotion-mlops-artifacts-619759452242-us-east-2 \
  --key models/emotion-cnn/v1/model.keras \
  --body models/emotion_cnn_model.keras \
  --content-type application/octet-stream \
  --server-side-encryption AES256 \
  --checksum-algorithm SHA256 \
  --metadata model-version=v1,model-sha256=9c8d773993e54ad78e7369ec5837890a8115933d288d4164eae370e9f0fbdb9b \
  --profile emotio-mlops-user \
  --region us-east-2

aws s3api put-object \
  --bucket emotion-mlops-artifacts-619759452242-us-east-2 \
  --key models/emotion-cnn/v1/metadata.json \
  --body models/emotion_cnn_model.metadata.json \
  --content-type application/json \
  --server-side-encryption AES256 \
  --checksum-algorithm SHA256 \
  --metadata model-version=v1,model-sha256=9c8d773993e54ad78e7369ec5837890a8115933d288d4164eae370e9f0fbdb9b \
  --profile emotio-mlops-user \
  --region us-east-2
```

Use a new logical version such as `v2` when model contents change. S3 bucket
versioning also preserves older physical object versions for recovery.

## Step 4: build and push the inference image

The POC stack from Step 2 creates the private ECR repository. Authenticate
Docker, then build the x86 image used by the EKS worker and push the immutable
release tag:

```bash
AWS_ACCOUNT_ID="$(aws sts get-caller-identity \
  --profile emotio-mlops-user \
  --query Account \
  --output text)"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.us-east-2.amazonaws.com"
IMAGE_URI="${ECR_REGISTRY}/emotion-api:v0.3.1"

aws ecr get-login-password \
  --profile emotio-mlops-user \
  --region us-east-2 \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

docker buildx build \
  --platform linux/amd64 \
  --tag "${IMAGE_URI}" \
  --push \
  .

aws ecr describe-images \
  --repository-name emotion-api \
  --image-ids imageTag=v0.3.1 \
  --profile emotio-mlops-user \
  --region us-east-2
```

Run these commands from the repository root. The Docker build requires
`models/emotion_cnn_model.keras` and its metadata file. Use a new tag whenever
the application or model changes; do not replace an existing release tag.

## Step 5: create the demo VPC and EKS cluster

The EKS stack is disabled by default to prevent accidental hourly charges. It
uses two public subnets, no NAT Gateway, one x86 Spot worker node, a 20 GiB
disk, and no load balancer. The Kubernetes API is limited to one public `/32`
address. This is appropriate for a short-lived learning environment, not a
production network design.

Before enabling EKS, an administrator must create and attach the execution
policy in `infra/iam/terraform-eks-policy.json`. The policy lets the project
user create only `emotion-mlops-*` IAM roles, requires a permissions boundary,
and limits which AWS-managed policies can be attached to those roles.

Run these two commands with an administrator profile, not with the project
user that receives the policy:

```bash
AWS_PROFILE=your-admin-profile aws iam create-policy \
  --policy-name EmotionMLOpsTerraformEksPolicy \
  --policy-document file://infra/iam/terraform-eks-policy.json

AWS_PROFILE=your-admin-profile aws iam attach-user-policy \
  --user-name realtime-emotional-mlops-user \
  --policy-arn arn:aws:iam::619759452242:policy/EmotionMLOpsTerraformEksPolicy
```

If no administrator CLI profile exists, perform the same one-time operation in
the IAM console: create a customer-managed policy from the JSON file and attach
it to `realtime-emotional-mlops-user`.

Confirm that the project user has the new read permissions:

```bash
aws ec2 describe-availability-zones \
  --query 'AvailabilityZones[?State==`available`].ZoneName' \
  --output table \
  --profile emotio-mlops-user \
  --region us-east-2

aws eks list-clusters \
  --profile emotio-mlops-user \
  --region us-east-2
```

Create and inspect an immutable Terraform plan. The public IP restriction is
important: do not replace it with `0.0.0.0/0`.

```bash
EKS_ADMIN_IP="$(curl -s https://checkip.amazonaws.com)"

AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc plan \
  -var=create_eks=true \
  -var="eks_admin_cidr=${EKS_ADMIN_IP}/32" \
  -out=tfplan

terraform -chdir=infra/terraform/environments/poc show tfplan

AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc apply tfplan
```

EKS creation usually takes 15-25 minutes. Configure `kubectl` and verify the
control plane, worker node, and managed add-ons after apply:

```bash
aws eks update-kubeconfig \
  --name emotion-mlops-poc \
  --region us-east-2 \
  --profile emotio-mlops-user

kubectl get nodes -o wide
kubectl get pods -n kube-system
terraform -chdir=infra/terraform/environments/poc output
```

The cluster uses EKS Pod Identity for the `emotion-api` service account. Its
role can read only the versioned emotion-model prefix in S3. Creating the core
cluster alone does not expose the FastAPI service.

## Step 6: deploy the public web demo

This separate Terraform state owns the Kubernetes application and the
short-lived public layer. It installs the AWS Load Balancer Controller, exposes
the internal Service through one internet-facing ALB, and creates a CloudFront
HTTPS URL. A generated secret origin header prevents normal direct requests to
the ALB. CloudFront caching is disabled because inference responses are unique.

The Terraform executor needs the existing EKS provisioning permissions, the
CloudFront and ELB permissions in
`infra/iam/terraform-platform-policy.json`, and the CloudWatch/SNS permissions
in `infra/iam/terraform-monitoring-policy.json`. These are intended to replace
broad `AdministratorAccess`; have an administrator create and attach the
scoped customer-managed policies if they are not already available.

Copy the non-secret example and set the notification address locally. The
resulting `terraform.tfvars` is ignored by Git:

```bash
cp infra/terraform/environments/poc-platform/terraform.tfvars.example \
  infra/terraform/environments/poc-platform/terraform.tfvars
```

Initialize, review, and apply:

```bash
AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc-platform init \
  -backend-config=backend.hcl

AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc-platform fmt -check

AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc-platform validate

AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc-platform plan \
  -out=tfplan

terraform -chdir=infra/terraform/environments/poc-platform show tfplan

AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc-platform apply tfplan
```

Get and test the generated HTTPS URL:

```bash
PUBLIC_URL="$(terraform \
  -chdir=infra/terraform/environments/poc-platform \
  output -raw public_url)"

curl "$PUBLIC_URL/health/ready"
curl "$PUBLIC_URL/v1/model"
curl -X POST \
  -F "file=@path/to/photo.jpg" \
  "$PUBLIC_URL/v1/predictions?largest_only=true"
```

Open `$PUBLIC_URL` in a browser and select **Start camera** or **Upload image**.
The browser requires permission before it can use the webcam and sends sampled
frames or the selected photo to FastAPI for inference.

### Verify monitoring and email alerts

The platform stack creates a CloudWatch dashboard for traffic, errors, and
latency, plus these alarms:

- ALB-generated 5xx count: at least one in five minutes
- FastAPI target 5xx count: at least one in five minutes
- p95 target response time: over three seconds for three of five minutes
- CloudFront 5xx error rate: over five percent for two of three minutes

Print the dashboard URL and open it in a browser:

```bash
terraform -chdir=infra/terraform/environments/poc-platform \
  output -raw cloudwatch_dashboard_url
```

AWS sends two SNS confirmation emails to the configured address. Confirm both:
one subscription is in `us-east-2` for ALB alarms, and the other is in
`us-east-1`, where CloudFront publishes its global metrics. Notifications do
not arrive until the subscriptions are confirmed.

Check the subscription and alarm state from the CLI:

```bash
AWS_PROFILE=emotio-mlops-user aws sns list-subscriptions-by-topic \
  --topic-arn arn:aws:sns:us-east-2:619759452242:emotion-mlops-poc-regional-alerts \
  --region us-east-2

AWS_PROFILE=emotio-mlops-user aws sns list-subscriptions-by-topic \
  --topic-arn arn:aws:sns:us-east-1:619759452242:emotion-mlops-poc-global-alerts \
  --region us-east-1

AWS_PROFILE=emotio-mlops-user aws cloudwatch describe-alarms \
  --alarm-name-prefix emotion-mlops-poc \
  --region us-east-2

AWS_PROFILE=emotio-mlops-user aws cloudwatch describe-alarms \
  --alarm-name-prefix emotion-mlops-poc \
  --region us-east-1
```

These are infrastructure/service-level signals. Centralized application logs
and model-quality signals such as prediction distribution and data drift are
separate future improvements.

The public POC layer adds roughly an ALB hourly charge plus LCUs and very small
CloudFront request/transfer charges. With the EKS control plane and one Spot
`t3.medium` node, budget approximately **$0.16-$0.17 per running hour** at light
demo traffic. This is an estimate, not a billing limit.

### Destroy the public demo and stop hourly charges

Order matters: destroy `poc-platform` first so Kubernetes can delete the ALB
while EKS still exists. Review both destroy plans before applying them.

If your public IP changed since EKS was created, update the cluster allowlist
first. Otherwise the Kubernetes provider will time out and Terraform will not
create `destroy.tfplan`:

```bash
export AWS_PROFILE=emotio-mlops-user
export AWS_REGION=us-east-2
export AWS_DEFAULT_REGION=us-east-2

EKS_ADMIN_IP="$(curl -fsS https://checkip.amazonaws.com | tr -d '\r\n')"

terraform -chdir=infra/terraform/environments/poc plan \
  -var=create_eks=true \
  -var="eks_admin_cidr=${EKS_ADMIN_IP}/32" \
  -out=update-access.tfplan

terraform -chdir=infra/terraform/environments/poc apply update-access.tfplan

aws eks update-kubeconfig \
  --name emotion-mlops-poc \
  --region us-east-2 \
  --profile emotio-mlops-user

kubectl get namespace emotion-api
```

Proceed only after the namespace command succeeds:

```bash
AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc-platform plan \
  -destroy \
  -out=destroy.tfplan

terraform -chdir=infra/terraform/environments/poc-platform \
  show destroy.tfplan

AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc-platform apply destroy.tfplan

AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc plan \
  -var=create_eks=false \
  -out=stop-eks.tfplan

terraform -chdir=infra/terraform/environments/poc show stop-eks.tfplan

AWS_PROFILE=emotio-mlops-user terraform \
  -chdir=infra/terraform/environments/poc apply stop-eks.tfplan
```

This preserves the inexpensive ECR images, versioned S3 model artifacts, and
remote Terraform state so the environment can be recreated later.

## Planned deployment order

1. Remote Terraform state.
2. Private S3 model-artifact bucket.
3. VPC and cost-optimized EKS cluster (implemented; disabled by default).
4. Kubernetes deployment of the ECR image (implemented).
5. ALB and CloudFront HTTPS public demo (implemented).
6. Operational CloudWatch monitoring and SNS alerting (implemented).
7. GitHub Actions validation CI (implemented); deployment CD and Argo CD
   remain future work.
8. MLflow model management.
9. Autoscaling and load testing.

Centralized application logs and model/data-drift monitoring remain future
work alongside those roadmap items.

Do not commit Terraform state, plan files, credentials, or local variable
files. Commit `.terraform.lock.hcl` so every environment uses the same provider
version.
