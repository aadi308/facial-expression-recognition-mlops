locals {
  controller_service_account = "aws-load-balancer-controller"
  permissions_boundary_arn   = "arn:aws:iam::aws:policy/PowerUserAccess"
}

data "aws_iam_policy_document" "controller_assume_role" {
  statement {
    actions = [
      "sts:AssumeRole",
      "sts:TagSession",
    ]

    principals {
      type        = "Service"
      identifiers = ["pods.eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "controller" {
  name                 = "${var.project_name}-${var.environment}-lbc-role"
  assume_role_policy   = data.aws_iam_policy_document.controller_assume_role.json
  permissions_boundary = local.permissions_boundary_arn
}

resource "aws_iam_role_policy" "controller" {
  name   = "aws-load-balancer-controller"
  role   = aws_iam_role.controller.id
  policy = file("${path.module}/lbc_iam_policy.json")
}

resource "aws_eks_pod_identity_association" "controller" {
  cluster_name    = var.cluster_name
  namespace       = "kube-system"
  service_account = local.controller_service_account
  role_arn        = aws_iam_role.controller.arn
}

resource "helm_release" "controller" {
  name       = "aws-load-balancer-controller"
  namespace  = "kube-system"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-load-balancer-controller"
  version    = var.load_balancer_controller_chart_version

  atomic          = true
  cleanup_on_fail = true
  timeout         = 600
  wait            = true

  values = [
    yamlencode({
      clusterName  = var.cluster_name
      region       = var.aws_region
      vpcId        = data.aws_eks_cluster.this.vpc_config[0].vpc_id
      replicaCount = 1
      serviceAccount = {
        create = true
        name   = local.controller_service_account
      }
      enableShield = false
      enableWaf    = false
      enableWafv2  = false
      resources = {
        requests = {
          cpu    = "50m"
          memory = "128Mi"
        }
        limits = {
          cpu    = "250m"
          memory = "256Mi"
        }
      }
      defaultTags = {
        Project     = var.project_name
        Environment = var.environment
        ManagedBy   = "terraform"
      }
    })
  ]

  # Keep both the Pod Identity association and its permissions alive until
  # every Ingress that depends on this controller has been deleted. Without
  # the policy dependency, Terraform may remove the inline policy early during
  # destroy, leaving ALB cleanup blocked on the Kubernetes finalizer.
  depends_on = [
    aws_eks_pod_identity_association.controller,
    aws_iam_role_policy.controller,
  ]
}
