output "vpc_id" {
  description = "ID of the demo VPC."
  value       = aws_vpc.this.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs used by the EKS control plane and demo nodes."
  value       = aws_subnet.public[*].id
}
