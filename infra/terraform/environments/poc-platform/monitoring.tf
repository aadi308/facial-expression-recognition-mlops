locals {
  regional_alarm_actions = var.alert_email == null ? [] : [aws_sns_topic.regional_alerts.arn]
  global_alarm_actions   = var.alert_email == null ? [] : [aws_sns_topic.global_alerts.arn]
}

data "aws_lb" "app" {
  name = "${var.project_name}-${var.environment}"

  depends_on = [kubernetes_ingress_v1.app]
}

resource "aws_sns_topic" "regional_alerts" {
  name = "${var.project_name}-${var.environment}-regional-alerts"
}

resource "aws_sns_topic_subscription" "regional_email" {
  count = var.alert_email == null ? 0 : 1

  topic_arn = aws_sns_topic.regional_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_sns_topic" "global_alerts" {
  provider = aws.global

  name = "${var.project_name}-${var.environment}-global-alerts"
}

resource "aws_sns_topic_subscription" "global_email" {
  provider = aws.global
  count    = var.alert_email == null ? 0 : 1

  topic_arn = aws_sns_topic.global_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_metric_alarm" "alb_load_balancer_5xx" {
  alarm_name          = "${var.project_name}-${var.environment}-alb-5xx"
  alarm_description   = "ALB-generated 5xx responses indicate routing or target availability failures."
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_ELB_5XX_Count"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"

  dimensions = {
    LoadBalancer = data.aws_lb.app.arn_suffix
  }

  alarm_actions = local.regional_alarm_actions
  ok_actions    = local.regional_alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "alb_target_5xx" {
  alarm_name          = "${var.project_name}-${var.environment}-target-5xx"
  alarm_description   = "The FastAPI target returned one or more 5xx responses in five minutes."
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"

  dimensions = {
    LoadBalancer = data.aws_lb.app.arn_suffix
  }

  alarm_actions = local.regional_alarm_actions
  ok_actions    = local.regional_alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "alb_latency" {
  alarm_name          = "${var.project_name}-${var.environment}-latency"
  alarm_description   = "The p95 ALB target response time exceeded three seconds for three minutes."
  namespace           = "AWS/ApplicationELB"
  metric_name         = "TargetResponseTime"
  extended_statistic  = "p95"
  period              = 60
  evaluation_periods  = 5
  datapoints_to_alarm = 3
  threshold           = 3
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  dimensions = {
    LoadBalancer = data.aws_lb.app.arn_suffix
  }

  alarm_actions = local.regional_alarm_actions
  ok_actions    = local.regional_alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "cloudfront_5xx" {
  provider = aws.global

  alarm_name          = "${var.project_name}-${var.environment}-cloudfront-5xx"
  alarm_description   = "More than five percent of CloudFront requests returned 5xx errors."
  namespace           = "AWS/CloudFront"
  metric_name         = "5xxErrorRate"
  statistic           = "Average"
  period              = 60
  evaluation_periods  = 3
  datapoints_to_alarm = 2
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  dimensions = {
    DistributionId = aws_cloudfront_distribution.app.id
    Region         = "Global"
  }

  alarm_actions = local.global_alarm_actions
  ok_actions    = local.global_alarm_actions
}

resource "aws_cloudwatch_dashboard" "app" {
  dashboard_name = "${var.project_name}-${var.environment}"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "text"
        x      = 0
        y      = 0
        width  = 24
        height = 2
        properties = {
          markdown = "# Emotion MLOps POC\nOperational health for CloudFront, the ALB, and the FastAPI inference target."
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 2
        width  = 12
        height = 6
        properties = {
          title  = "Traffic and target errors"
          region = var.aws_region
          view   = "timeSeries"
          period = 300
          stat   = "Sum"
          metrics = [
            ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", data.aws_lb.app.arn_suffix, { label = "Requests" }],
            ["AWS/ApplicationELB", "HTTPCode_Target_5XX_Count", "LoadBalancer", data.aws_lb.app.arn_suffix, { label = "Target 5xx" }],
            ["AWS/ApplicationELB", "HTTPCode_ELB_5XX_Count", "LoadBalancer", data.aws_lb.app.arn_suffix, { label = "ALB 5xx" }],
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 2
        width  = 12
        height = 6
        properties = {
          title  = "Inference service latency"
          region = var.aws_region
          view   = "timeSeries"
          period = 60
          metrics = [
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", data.aws_lb.app.arn_suffix, { stat = "p50", label = "p50" }],
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", data.aws_lb.app.arn_suffix, { stat = "p95", label = "p95" }],
            ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", data.aws_lb.app.arn_suffix, { stat = "p99", label = "p99" }],
          ]
          yAxis = {
            left = { min = 0, label = "seconds" }
          }
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 8
        width  = 24
        height = 6
        properties = {
          title  = "CloudFront requests and 5xx rate"
          region = "us-east-1"
          view   = "timeSeries"
          period = 300
          metrics = [
            ["AWS/CloudFront", "Requests", "DistributionId", aws_cloudfront_distribution.app.id, "Region", "Global", { stat = "Sum", label = "Requests", yAxis = "left" }],
            ["AWS/CloudFront", "5xxErrorRate", "DistributionId", aws_cloudfront_distribution.app.id, "Region", "Global", { stat = "Average", label = "5xx error rate", yAxis = "right" }],
          ]
          yAxis = {
            right = { min = 0, max = 100, label = "percent" }
          }
        }
      },
    ]
  })
}
