"""Shared notebook helpers for Airbnb Market Intelligence EDA.

Provides:
  - AirbnbDB: DuckDB connection manager returning Pandas DataFrames
  - Plot styling with Airbnb-inspired colour palette
  - Business insight formatting for markdown cells
  - City centre coordinates for geographic analysis
  - Currency formatting utilities

Usage in notebooks:
    import sys; sys.path.insert(0, "..")
    from notebooks.helpers import AirbnbDB, set_airbnb_style, business_insight

    db = AirbnbDB()
    df = db.query("SELECT * FROM fact_listing_snapshot LIMIT 10")
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ===================================================================
# Paths
# ===================================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DB_PATH = _PROJECT_ROOT / "data" / "airbnb.duckdb"
_SQL_DIR = _PROJECT_ROOT / "sql"
_DATA_DIR = _PROJECT_ROOT / "data"
_CONFIG_DIR = _PROJECT_ROOT / "config"

# ===================================================================
# Airbnb brand palette
# ===================================================================

AIRBNB_PALETTE = [
    "#FF5A5F",  # Rausch (primary red/coral)
    "#00A699",  # Babu (teal)
    "#FC642D",  # Arches (orange)
    "#484848",  # Hof (dark grey)
    "#767676",  # Foggy (medium grey)
    "#E8A838",  # Gold
    "#7B0051",  # Plum
    "#C4D600",  # Lime
    "#3B5998",  # Slate blue
    "#D63ADF",  # Violet
]

AIRBNB_DIVERGING = ["#FF5A5F", "#FFB3B5", "#F5F5F5", "#A8DCD8", "#00A699"]

# City centres for distance calculations
CITY_CENTRES: dict[str, tuple[float, float]] = {
    "paris": (48.8566, 2.3522),  # Notre-Dame
    "new_york_city": (40.7580, -73.9855),  # Times Square
    "london": (51.5074, -0.1278),  # Trafalgar Square
    "barcelona": (41.3874, 2.1686),  # Plaça de Catalunya
}

# City display names
CITY_DISPLAY: dict[str, str] = {
    "paris": "Paris",
    "new_york_city": "New York City",
    "london": "London",
}


# ===================================================================
# DuckDB Connection Manager
# ===================================================================


class AirbnbDB:
    """Context-manager for read-only access to the Airbnb DuckDB star schema.

    Returns query results as Pandas DataFrames for compatibility with
    matplotlib, seaborn, and plotly.

    Usage:
        db = AirbnbDB()
        df = db.query("SELECT * FROM dim_city")

        # Or as context manager:
        with AirbnbDB() as db:
            df = db.query("SELECT * FROM fact_listing_snapshot LIMIT 5")
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DB_PATH
        self._con: duckdb.DuckDBPyConnection | None = None

    def _ensure_connection(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            if not self._db_path.exists():
                raise FileNotFoundError(
                    f"DuckDB database not found: {self._db_path}\n"
                    f"Run the pipeline first: python main.py model --cities paris,london,new_york_city"
                )
            self._con = duckdb.connect(str(self._db_path), read_only=True)
        return self._con

    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return a Pandas DataFrame."""
        con = self._ensure_connection()
        return con.execute(sql).fetchdf()

    def query_named(self, name: str) -> pd.DataFrame:
        """Run a named query from sql/analytical_queries.sql."""
        queries = self._parse_named_queries()
        if name not in queries:
            available = ", ".join(sorted(queries.keys()))
            raise KeyError(f"Unknown query '{name}'. Available: {available}")
        return self.query(queries[name])

    def table_info(self) -> pd.DataFrame:
        """Return row counts for all star-schema tables."""
        con = self._ensure_connection()
        tables = [
            "dim_date",
            "dim_city",
            "dim_host",
            "dim_property",
            "dim_neighbourhood",
            "dim_reviewer",
            "fact_listing_snapshot",
            "fact_calendar",
            "fact_review",
        ]
        rows = []
        for t in tables:
            try:
                count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                rows.append({"table": t, "row_count": count})
            except Exception:
                rows.append({"table": t, "row_count": -1})
        return pd.DataFrame(rows)

    def _parse_named_queries(self) -> dict[str, str]:
        """Parse -- name: blocks from analytical_queries.sql."""
        sql_path = _SQL_DIR / "analytical_queries.sql"
        text = sql_path.read_text(encoding="utf-8")
        matches = list(re.finditer(r"^--\s*name:\s*(\w+)\s*$", text, flags=re.MULTILINE))
        queries: dict[str, str] = {}
        for idx, match in enumerate(matches):
            name = match.group(1)
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = text[start:end].strip().rstrip(";")
            if body:
                queries[name] = body
        return queries

    def close(self) -> None:
        """Close the database connection."""
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> AirbnbDB:
        self._ensure_connection()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


# ===================================================================
# Enriched Parquet loader (for columns not in star schema)
# ===================================================================


def load_enriched_master(city: str | None = None) -> pd.DataFrame:
    """Load enriched master listings Parquet as Pandas DataFrame.

    Use this for columns not in the star schema (e.g., raw amenities,
    text fields). For most analyses, prefer db.query() instead.

    Args:
        city: City key (e.g., 'paris'). If None, loads unified master.

    Returns:
        Pandas DataFrame.
    """
    import polars as pl

    enriched_dir = _DATA_DIR / "enriched"
    if city:
        path = enriched_dir / f"{city}_master_listings.parquet"
    else:
        path = enriched_dir / "unified_master_listings.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Enriched master not found: {path}\nRun: python main.py enrich --city {city or 'all'}"
        )

    return pl.read_parquet(path).to_pandas()


def load_raw_geojson(city: str) -> dict:
    """Load the raw GeoJSON for a city's neighbourhoods."""
    import json

    raw_dir = _DATA_DIR / "raw" / city
    geojson_path = raw_dir / "neighbourhoods.geojson"

    if not geojson_path.exists():
        raise FileNotFoundError(
            f"GeoJSON not found: {geojson_path}\nRun: python main.py download --city {city}"
        )

    with open(geojson_path, encoding="utf-8") as f:
        return json.load(f)


# ===================================================================
# Plot Styling
# ===================================================================


def set_airbnb_style(dark: bool = False) -> None:
    """Configure matplotlib and seaborn with Airbnb-inspired styling.

    Args:
        dark: If True, use a dark background theme.
    """
    base_style = "dark_background" if dark else "seaborn-v0_8-whitegrid"
    plt.style.use(base_style)

    sns.set_palette(AIRBNB_PALETTE)

    plt.rcParams.update(
        {
            # Typography
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "Arial"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            # Layout
            "figure.figsize": (12, 6),
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "figure.facecolor": "#1a1a2e" if dark else "white",
            "axes.facecolor": "#16213e" if dark else "#fafafa",
            "axes.edgecolor": "#444" if dark else "#cccccc",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            # Colours
            "text.color": "#e0e0e0" if dark else "#484848",
            "axes.labelcolor": "#e0e0e0" if dark else "#484848",
            "xtick.color": "#aaa" if dark else "#767676",
            "ytick.color": "#aaa" if dark else "#767676",
        }
    )


# ===================================================================
# Business Insight Formatter
# ===================================================================


def business_insight(
    title: str,
    finding: str,
    implication: str,
    action: str,
) -> str:
    """Generate a formatted business insight for display in a markdown cell.

    Copy the returned string into a Markdown cell using:
        from IPython.display import Markdown, display
        display(Markdown(business_insight(...)))

    Args:
        title: Short insight title.
        finding: Statistical observation.
        implication: What it means for stakeholders.
        action: Recommended action.

    Returns:
        Formatted markdown string.
    """
    return textwrap.dedent(f"""\
    ### 📊 Business Insight: {title}

    **Finding:** {finding}

    **Business Implication:** {implication}

    **Recommended Action:** {action}
    """)


# ===================================================================
# Formatting Utilities
# ===================================================================


def fmt_currency(value: float, symbol: str = "$", decimals: int = 0) -> str:
    """Format a number as currency: $1,250."""
    if pd.isna(value):
        return "N/A"
    return f"{symbol}{value:,.{decimals}f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    """Format a number as percentage: 85.3%."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"


def fmt_count(value: int | float) -> str:
    """Format a number with thousands separators: 1,234,567."""
    if pd.isna(value):
        return "N/A"
    return f"{int(value):,}"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate distance in km using equirectangular projection.

    Accurate enough at city scale (< 0.5% error within 50km).
    """
    import math

    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111.32
    dy = (lat2 - lat1) * 111.32
    return math.sqrt(dx * dx + dy * dy)
