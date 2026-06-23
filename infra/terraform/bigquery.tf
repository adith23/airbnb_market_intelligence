resource "google_bigquery_dataset" "airbnb_data" {
  dataset_id                  = "airbnb_market_intelligence_${var.environment}"
  friendly_name               = "Airbnb Market Intelligence Dataset"
  description                 = "Contains raw, staging, and modeled facts/dims for Airbnb data"
  location                    = var.region
  default_table_expiration_ms = null # Tables don't expire
}
