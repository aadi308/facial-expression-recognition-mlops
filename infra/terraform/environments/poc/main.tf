data "aws_caller_identity" "current" {}

module "ecr" {
  source = "../../modules/ecr"

  repository_name     = "emotion-api"
  release_tag_prefix  = "v"
  release_image_count = 5
}

module "model_artifacts" {
  source = "../../modules/artifact_bucket"

  bucket_name = "${var.project_name}-artifacts-${data.aws_caller_identity.current.account_id}-${var.aws_region}"
}

module "network" {
  source = "../../modules/network"
  count  = var.create_eks ? 1 : 0

  name     = "${var.project_name}-${var.environment}"
  vpc_cidr = var.vpc_cidr
}

module "eks" {
  source = "../../modules/eks"
  count  = var.create_eks ? 1 : 0

  cluster_name       = "${var.project_name}-${var.environment}"
  kubernetes_version = var.kubernetes_version
  subnet_ids         = module.network[0].public_subnet_ids
  admin_cidr         = var.eks_admin_cidr
  model_bucket_arn   = module.model_artifacts.bucket_arn
}
