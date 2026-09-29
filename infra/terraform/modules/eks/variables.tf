variable "cluster_name" {
  description = "Name of the EKS cluster."
  type        = string
}

variable "kubernetes_version" {
  description = "EKS Kubernetes minor version."
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs used by the control plane and managed node group."
  type        = list(string)
}

variable "admin_cidr" {
  description = "Single trusted public IPv4 CIDR allowed to reach the Kubernetes API."
  type        = string

  validation {
    condition     = can(cidrhost(var.admin_cidr, 0)) && endswith(var.admin_cidr, "/32")
    error_message = "admin_cidr must be a valid single-host CIDR ending in /32."
  }
}

variable "model_bucket_arn" {
  description = "ARN of the S3 bucket containing model artifacts."
  type        = string
}

variable "model_object_prefix" {
  description = "S3 object prefix that the inference workload may read."
  type        = string
  default     = "models/emotion-cnn/"
}

variable "node_instance_types" {
  description = "Compatible x86 instance types used by the Spot node group."
  type        = list(string)
  default     = ["t3.medium", "t3a.medium"]
}

variable "tags" {
  description = "Additional tags applied to EKS resources."
  type        = map(string)
  default     = {}
}
