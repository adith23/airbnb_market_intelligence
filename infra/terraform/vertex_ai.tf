resource "google_artifact_registry_repository" "ml_models" {
  location      = var.region
  repository_id = "airbnb-ml-models-${var.environment}"
  description   = "Docker repository for Vertex AI Custom Training"
  format        = "DOCKER"
}
