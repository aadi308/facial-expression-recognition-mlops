output "cluster_name" {
  description = "Name of the EKS cluster."
  value       = aws_eks_cluster.this.name
}

output "cluster_endpoint" {
  description = "Kubernetes API endpoint."
  value       = aws_eks_cluster.this.endpoint
}

output "application_role_arn" {
  description = "Pod Identity role used by the inference API to read its model."
  value       = aws_iam_role.application.arn
}

output "node_role_arn" {
  description = "IAM role used by EKS worker nodes."
  value       = aws_iam_role.node.arn
}
