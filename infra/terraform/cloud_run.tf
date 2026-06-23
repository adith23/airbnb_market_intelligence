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
    execution_environment = "EXECUTION_ENVIRONMENT_GEN2"
    containers {
      # Use a placeholder image initially. CI/CD will update it.
      image = "us-docker.pkg.dev/cloudrun/container/hello"
      
      volume_mounts {
        name       = "gcs-fuse"
        mount_path = "/app/shared_data"
      }
      
      env {
        name  = "AIRFLOW_DATA_DIR"
        value = "/app/shared_data"
      }
    }
    
    volumes {
      name = "gcs-fuse"
      gcs {
        bucket = google_storage_bucket.processed_data.name
        read_only = true
      }
    }
    service_account = var.service_account_email
  }

  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
    ]
  }
}
