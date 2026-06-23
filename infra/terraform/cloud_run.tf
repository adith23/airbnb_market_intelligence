resource "google_artifact_registry_repository" "app_repo" {
  location      = var.region
  repository_id = "airbnb-app-repo-${var.environment}"
  description   = "Docker repository for Dashboard app"
  format        = "DOCKER"
}

resource "google_cloud_run_v2_service" "dashboard" {
  name     = "airbnb-dashboard-${var.environment}"
  location = var.region

  template {
    containers {
      # Use a placeholder image initially. CI/CD will update it.
      image = "us-docker.pkg.dev/cloudrun/container/hello"
    }
  }

  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
    ]
  }
}
