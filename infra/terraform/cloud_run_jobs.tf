# =========================================================
# Cloud Run Jobs (Serverless Alternatives to Airflow)
# =========================================================

# Data Pipeline Job
resource "google_cloud_run_v2_job" "data_pipeline" {
  provider = google-beta
  name     = "airbnb-data-pipeline-${var.environment}"
  location = var.region

  template {
    template {
      execution_environment = "EXECUTION_ENVIRONMENT_GEN2"
      containers {
        image = "us-docker.pkg.dev/cloudrun/container/hello" # Replaced by CI/CD
        command = ["python", "main.py", "run-pipeline-all"]
        
        resources {
          limits = {
            memory = "8Gi"
            cpu    = "2"
          }
        }
        
        env {
          name  = "ENV"
          value = var.environment
        }
        env {
          name  = "GCP_PROJECT_ID"
          value = var.project_id
        }
        env {
          name  = "GCP_REGION"
          value = var.region
        }
        env {
          name  = "AIRFLOW_DATA_DIR"
          value = "/app/shared_data"
        }
        
        # Native GCS FUSE integration for Cloud Run Jobs
        volume_mounts {
          name       = "gcs-fuse"
          mount_path = "/app/shared_data"
        }
      }
      
      volumes {
        name = "gcs-fuse"
        gcs {
          bucket = google_storage_bucket.processed_data.name
          read_only = false
        }
      }
      service_account = var.service_account_email
    }
  }

  lifecycle {
    ignore_changes = [
      template[0].template[0].containers[0].image,
    ]
  }
}

# ML Pipeline Job
resource "google_cloud_run_v2_job" "ml_pipeline" {
  provider = google-beta
  name     = "airbnb-ml-pipeline-${var.environment}"
  location = var.region

  template {
    template {
      execution_environment = "EXECUTION_ENVIRONMENT_GEN2"
      containers {
        image = "us-docker.pkg.dev/cloudrun/container/hello" # Replaced by CI/CD
        
        # Executes the full ML Orchestrator script
        command = ["python", "-c", "from src.platform.mlops.orchestrator import run_ml_pipeline; run_ml_pipeline('config/ml_config.yaml')"]
        
        resources {
          limits = {
            memory = "8Gi"
            cpu    = "2"
          }
        }
        
        env {
          name  = "ENV"
          value = var.environment
        }
        env {
          name  = "GCP_PROJECT_ID"
          value = var.project_id
        }
        env {
          name  = "GCP_REGION"
          value = var.region
        }
        env {
          name  = "AIRFLOW_DATA_DIR"
          value = "/app/shared_data"
        }
        
        volume_mounts {
          name       = "gcs-fuse"
          mount_path = "/app/shared_data"
        }
      }
      
      volumes {
        name = "gcs-fuse"
        gcs {
          bucket = google_storage_bucket.processed_data.name
          read_only = false
        }
      }
      service_account = var.service_account_email
    }
  }

  lifecycle {
    ignore_changes = [
      template[0].template[0].containers[0].image,
    ]
  }
}

# =========================================================
# Cloud Scheduler Triggers
# =========================================================

# Trigger Data Pipeline Weekly (e.g., Sunday at 2 AM)
resource "google_cloud_scheduler_job" "trigger_data_pipeline" {
  name             = "trigger-data-pipeline-${var.environment}"
  description      = "Triggers the AirBnB Data Pipeline"
  schedule         = "0 2 * * 0"
  time_zone        = "UTC"
  
  http_target {
    http_method = "POST"
    uri         = "https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/${google_cloud_run_v2_job.data_pipeline.name}:run"
    oauth_token {
      service_account_email = var.service_account_email
    }
  }
}

# Trigger ML Pipeline Weekly (e.g., Sunday at 4 AM)
resource "google_cloud_scheduler_job" "trigger_ml_pipeline" {
  name             = "trigger-ml-pipeline-${var.environment}"
  description      = "Triggers the AirBnB ML Pipeline"
  schedule         = "0 4 * * 0"
  time_zone        = "UTC"
  
  http_target {
    http_method = "POST"
    uri         = "https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/${google_cloud_run_v2_job.ml_pipeline.name}:run"
    oauth_token {
      service_account_email = var.service_account_email
    }
  }
}
