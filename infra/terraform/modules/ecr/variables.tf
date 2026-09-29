variable "repository_name" {
  description = "Name of the private ECR repository."
  type        = string
}

variable "release_tag_prefix" {
  description = "Tag prefix used to identify release images."
  type        = string
  default     = "v"
}

variable "release_image_count" {
  description = "Number of recent release images retained by ECR."
  type        = number
  default     = 5

  validation {
    condition     = var.release_image_count >= 1
    error_message = "release_image_count must be at least 1."
  }
}

variable "tags" {
  description = "Tags applied to the ECR repository."
  type        = map(string)
  default     = {}
}
