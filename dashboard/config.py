"""Configuration settings for the Streamlit Dashboard."""

from pathlib import Path

import os

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
AIRFLOW_DATA_DIR = os.environ.get("AIRFLOW_DATA_DIR")

if AIRFLOW_DATA_DIR:
    # Production path via GCS FUSE
    DB_PATH = str(Path(AIRFLOW_DATA_DIR) / "airbnb_data" / "airbnb.duckdb")
    MODELS_DIR = Path(AIRFLOW_DATA_DIR) / "airbnb_data" / "models"
else:
    # Local fallback
    DB_PATH = str(PROJECT_ROOT / "data" / "airbnb.duckdb")
    MODELS_DIR = PROJECT_ROOT / "data" / "models"

# UI Settings
PAGE_TITLE = "Airbnb Market Intelligence"
PAGE_ICON = "🏠"
LAYOUT = "wide"

# Styling
PRIMARY_COLOR = "#FF5A5F"  # Airbnb Red
