variable "aws_region" {
  description = "AWS region containing the EKS cluster."
  type        = string
  default     = "us-east-2"
}

variable "aws_profile" {
  description = "Local AWS CLI profile used by Terraform and Kubernetes auth."
  type        = string
  default     = "emotio-mlops-user"
}

variable "project_name" {
  description = "Project identifier used in resource names and tags."
  type        = string
  default     = "emotion-mlops"
}

variable "environment" {
  description = "Deployment environment name."
  type        = string
  default     = "poc"
}

variable "cluster_name" {
  description = "Existing EKS cluster that hosts the public demo."
  type        = string
  default     = "emotion-mlops-poc"
}

variable "image" {
  description = "Immutable ECR image deployed to the inference Pod."
  type        = string
  default     = "619759452242.dkr.ecr.us-east-2.amazonaws.com/emotion-api:v0.3.1"
}

variable "load_balancer_controller_chart_version" {
  description = "Pinned AWS Load Balancer Controller Helm chart version."
  type        = string
  default     = "3.5.0"
}

variable "alert_email" {
  description = "Email endpoint for regional and global SNS alarm notifications."
  type        = string
  default     = null
  sensitive   = true

  validation {
    condition = var.alert_email == null || can(
      regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.alert_email)
    )
    error_message = "alert_email must be null or a valid email address."
  }
}
