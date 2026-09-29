locals {
  app_name      = "emotion-api"
  app_namespace = "emotion-api"
  app_labels = {
    "app.kubernetes.io/name"      = "emotion-api"
    "app.kubernetes.io/component" = "inference"
  }
}

resource "random_password" "origin_header" {
  length  = 32
  special = false
}

resource "kubernetes_namespace_v1" "app" {
  metadata {
    name = local.app_namespace
    labels = {
      "pod-security.kubernetes.io/enforce" = "restricted"
      "pod-security.kubernetes.io/audit"   = "restricted"
      "pod-security.kubernetes.io/warn"    = "restricted"
    }
  }
}

resource "kubernetes_service_account_v1" "app" {
  automount_service_account_token = false

  metadata {
    name      = local.app_name
    namespace = kubernetes_namespace_v1.app.metadata[0].name
    labels    = local.app_labels
  }
}

resource "kubernetes_deployment_v1" "app" {
  metadata {
    name      = local.app_name
    namespace = kubernetes_namespace_v1.app.metadata[0].name
    labels    = local.app_labels
  }

  spec {
    replicas               = 1
    revision_history_limit = 2

    strategy {
      type = "RollingUpdate"
      rolling_update {
        max_surge       = "0"
        max_unavailable = "1"
      }
    }

    selector {
      match_labels = {
        "app.kubernetes.io/name" = local.app_name
      }
    }

    template {
      metadata {
        annotations = {
          "prometheus.io/path"   = "/metrics"
          "prometheus.io/port"   = "8000"
          "prometheus.io/scrape" = "true"
        }
        labels = local.app_labels
      }

      spec {
        automount_service_account_token  = false
        enable_service_links             = false
        service_account_name             = kubernetes_service_account_v1.app.metadata[0].name
        termination_grace_period_seconds = 30

        security_context {
          run_as_non_root = true
          seccomp_profile {
            type = "RuntimeDefault"
          }
        }

        container {
          name              = "api"
          image             = var.image
          image_pull_policy = "IfNotPresent"

          port {
            name           = "http"
            container_port = 8000
            protocol       = "TCP"
          }

          env {
            name  = "HOME"
            value = "/tmp"
          }

          resources {
            requests = {
              cpu    = "250m"
              memory = "512Mi"
            }
            limits = {
              cpu    = "1"
              memory = "1536Mi"
            }
          }

          security_context {
            allow_privilege_escalation = false
            read_only_root_filesystem  = true
            run_as_non_root            = true
            run_as_user                = 10001
            run_as_group               = 10001
            capabilities {
              drop = ["ALL"]
            }
          }

          startup_probe {
            http_get {
              path = "/health/ready"
              port = "http"
            }
            period_seconds    = 5
            timeout_seconds   = 2
            failure_threshold = 24
          }

          readiness_probe {
            http_get {
              path = "/health/ready"
              port = "http"
            }
            period_seconds    = 10
            timeout_seconds   = 2
            failure_threshold = 3
          }

          liveness_probe {
            http_get {
              path = "/health/live"
              port = "http"
            }
            period_seconds    = 20
            timeout_seconds   = 2
            failure_threshold = 3
          }

          volume_mount {
            name       = "temporary-files"
            mount_path = "/tmp"
          }
        }

        volume {
          name = "temporary-files"
          empty_dir {}
        }
      }
    }
  }

  wait_for_rollout = true
}

resource "kubernetes_service_v1" "app" {
  metadata {
    name      = local.app_name
    namespace = kubernetes_namespace_v1.app.metadata[0].name
    labels    = local.app_labels
  }

  spec {
    type = "ClusterIP"
    selector = {
      "app.kubernetes.io/name" = local.app_name
    }

    port {
      name        = "http"
      port        = 80
      target_port = "http"
      protocol    = "TCP"
    }
  }
}

resource "kubernetes_ingress_v1" "app" {
  wait_for_load_balancer = true

  metadata {
    name      = local.app_name
    namespace = kubernetes_namespace_v1.app.metadata[0].name
    labels    = local.app_labels
    annotations = {
      "alb.ingress.kubernetes.io/scheme"                  = "internet-facing"
      "alb.ingress.kubernetes.io/target-type"             = "ip"
      "alb.ingress.kubernetes.io/load-balancer-name"      = "${var.project_name}-${var.environment}"
      "alb.ingress.kubernetes.io/listen-ports"            = jsonencode([{ HTTP = 80 }])
      "alb.ingress.kubernetes.io/healthcheck-path"        = "/health/ready"
      "alb.ingress.kubernetes.io/healthcheck-port"        = "traffic-port"
      "alb.ingress.kubernetes.io/success-codes"           = "200"
      "alb.ingress.kubernetes.io/target-group-attributes" = "deregistration_delay.timeout_seconds=30"
      "alb.ingress.kubernetes.io/load-balancer-attributes" = join(",", [
        "deletion_protection.enabled=false",
        "idle_timeout.timeout_seconds=60",
      ])
      "alb.ingress.kubernetes.io/tags" = join(",", [
        "Project=${var.project_name}",
        "Environment=${var.environment}",
        "ManagedBy=terraform",
      ])
      "alb.ingress.kubernetes.io/conditions.${local.app_name}" = jsonencode([
        {
          field = "http-header"
          httpHeaderConfig = {
            httpHeaderName = "X-Origin-Verify"
            values         = [random_password.origin_header.result]
          }
        }
      ])
    }
  }

  spec {
    ingress_class_name = "alb"

    rule {
      http {
        path {
          path      = "/"
          path_type = "Prefix"
          backend {
            service {
              name = kubernetes_service_v1.app.metadata[0].name
              port {
                number = 80
              }
            }
          }
        }
      }
    }
  }

  timeouts {
    create = "20m"
    delete = "15m"
  }

  depends_on = [helm_release.controller]
}
