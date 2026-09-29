output "public_url" {
  description = "Public HTTPS webcam demo URL."
  value       = "https://${aws_cloudfront_distribution.app.domain_name}"
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution identifier."
  value       = aws_cloudfront_distribution.app.id
}

output "alb_hostname" {
  description = "ALB origin hostname; direct requests are rejected without the secret origin header."
  value       = kubernetes_ingress_v1.app.status[0].load_balancer[0].ingress[0].hostname
}

output "deployed_image" {
  description = "Immutable ECR image running in EKS."
  value       = var.image
}

output "cloudwatch_dashboard_url" {
  description = "AWS Console URL for the operational monitoring dashboard."
  value       = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards/dashboard/${aws_cloudwatch_dashboard.app.dashboard_name}"
}

output "regional_alert_topic_arn" {
  description = "SNS topic used by regional ALB alarms."
  value       = aws_sns_topic.regional_alerts.arn
}

output "global_alert_topic_arn" {
  description = "SNS topic used by the global CloudFront alarm."
  value       = aws_sns_topic.global_alerts.arn
}
