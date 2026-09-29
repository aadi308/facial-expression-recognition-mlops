variable "aws_region" {
  description = "AWS region containing the Terraform state bucket."
  type        = string
  default     = "us-east-2"
}

variable "aws_profile" {
  description = "Local AWS CLI profile used to bootstrap Terraform."
  type        = string
  default     = "emotio-mlops-user"
}

variable "project_name" {
  description = "Project name used for resource names and tags."
  type        = string
  default     = "emotion-mlops"
}

variable "noncurrent_state_retention_days" {
  description = "Days to retain old state-file versions for recovery."
  type        = number
  default     = 90

  validation {
    condition     = var.noncurrent_state_retention_days >= 30
    error_message = "State history must be retained for at least 30 days."
  }
}
