output "bucket_arn" {
  description = "ARN of the private model-artifact bucket."
  value       = aws_s3_bucket.this.arn
}

output "bucket_name" {
  description = "Name of the private model-artifact bucket."
  value       = aws_s3_bucket.this.id
}
