output "ecr_repository_arn" {
  description = "ARN of the API container repository."
  value       = module.ecr.repository_arn
}

output "ecr_repository_url" {
  description = "Docker registry URL used to push and deploy API images."
  value       = module.ecr.repository_url
}

output "model_artifact_bucket_arn" {
  description = "ARN of the private bucket containing versioned model artifacts."
  value       = module.model_artifacts.bucket_arn
}

output "model_artifact_bucket_name" {
  description = "Name of the private bucket containing versioned model artifacts."
  value       = module.model_artifacts.bucket_name
}

output "eks_cluster_name" {
  description = "Name used to configure kubectl after Terraform apply."
  value       = try(module.eks[0].cluster_name, null)
}

output "eks_cluster_endpoint" {
  description = "Restricted Kubernetes API endpoint."
  value       = try(module.eks[0].cluster_endpoint, null)
}

output "emotion_api_role_arn" {
  description = "Least-privilege Pod Identity role for model downloads."
  value       = try(module.eks[0].application_role_arn, null)
}

output "vpc_id" {
  description = "ID of the isolated demo VPC."
  value       = try(module.network[0].vpc_id, null)
}
