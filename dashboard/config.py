"""Configuration settings for the Streamlit Dashboard."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = str(PROJECT_ROOT / "data" / "airbnb.duckdb")
MODELS_DIR = PROJECT_ROOT / "data" / "models"

# UI Settings
PAGE_TITLE = "Airbnb Market Intelligence"
PAGE_ICON = "🏠"
LAYOUT = "wide"

# Styling
PRIMARY_COLOR = "#FF5A5F"  # Airbnb Red
