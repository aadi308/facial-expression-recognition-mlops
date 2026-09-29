variable "aws_region" {
  description = "AWS region used by the POC environment."
  type        = string
  default     = "us-east-2"
}

variable "aws_profile" {
  description = "Local AWS CLI profile used to run Terraform."
  type        = string
  default     = "emotio-mlops-user"
}

variable "project_name" {
  description = "Project tag applied to AWS resources."
  type        = string
  default     = "emotion-mlops"
}

variable "environment" {
  description = "Deployment environment name."
  type        = string
  default     = "poc"
}

variable "vpc_cidr" {
  description = "IPv4 CIDR for the isolated demo VPC."
  type        = string
  default     = "10.20.0.0/16"
}

variable "kubernetes_version" {
  description = "EKS Kubernetes minor version under standard support."
  type        = string
  default     = "1.35"
}

variable "create_eks" {
  description = "Cost-safety switch: create or remove the demo VPC and EKS cluster."
  type        = bool
  default     = false
}

variable "eks_admin_cidr" {
  description = "Your current public IPv4 address as a /32 CIDR; no default is allowed."
  type        = string
  sensitive   = true
  default     = null

  validation {
    condition = !var.create_eks || (
      var.eks_admin_cidr != null &&
      can(cidrhost(var.eks_admin_cidr, 0)) &&
      endswith(var.eks_admin_cidr, "/32")
    )
    error_message = "When create_eks is true, eks_admin_cidr must be a valid single-host CIDR ending in /32."
  }
}
