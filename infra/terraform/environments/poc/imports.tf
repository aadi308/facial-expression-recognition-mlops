# These imports adopt resources created during the initial ECR learning step.
# Terraform will not recreate them or remove the images already stored in ECR.
import {
  to = module.ecr.aws_ecr_repository.this
  id = "emotion-api"
}

import {
  to = module.ecr.aws_ecr_lifecycle_policy.this
  id = "emotion-api"
}
