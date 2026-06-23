resource "google_storage_bucket" "raw_data" {
  name          = "airbnb-raw-data-${var.project_id}-${var.environment}"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
}

resource "google_storage_bucket" "processed_data" {
  name          = "airbnb-processed-data-${var.project_id}-${var.environment}"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true
}
