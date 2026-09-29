variable "bucket_name" {
  description = "Globally unique name of the private model-artifact bucket."
  type        = string
}

variable "noncurrent_version_retention_days" {
  description = "Days to retain older versions of model artifacts."
  type        = number
  default     = 90

  validation {
    condition     = var.noncurrent_version_retention_days >= 30
    error_message = "Artifact history must be retained for at least 30 days."
  }
}

variable "tags" {
  description = "Additional tags applied to the artifact bucket."
  type        = map(string)
  default     = {}
}
