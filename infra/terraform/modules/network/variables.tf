variable "name" {
  description = "Name prefix for networking resources."
  type        = string
}

variable "vpc_cidr" {
  description = "IPv4 CIDR allocated to the demo VPC."
  type        = string
  default     = "10.20.0.0/16"

  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "vpc_cidr must be a valid IPv4 CIDR."
  }
}

variable "tags" {
  description = "Additional tags applied to networking resources."
  type        = map(string)
  default     = {}
}
