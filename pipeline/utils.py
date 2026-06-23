"""Shared utility functions for data parsing and cleaning.

This module provides idempotent transformation functions used across
the pipeline for parsing Inside Airbnb's raw data formats into clean,
typed values. Every function is:
  - Pure (no side effects)
  - Null-safe (handles None/NaN gracefully)
  - Documented with examples

These are consumed by the profiler, validator, and harmonizer modules.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from ast import literal_eval
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex patterns (module-level for performance)
# ---------------------------------------------------------------------------
CURRENCY_SYMBOL_RE = re.compile(r"[\$€£,]")
NUMERIC_EXTRACT_RE = re.compile(r"(\d+\.?\d*)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
HTML_ENTITY_RE = re.compile(r"&\w+;")

# Airbnb boolean encoding
BOOLEAN_MAP: dict[str, bool] = {"t": True, "f": False}

# Project root resolved relative to this file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ===================================================================
# Logging
# ===================================================================


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging for the entire pipeline.

    Args:
        level: Python logging level (default: INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ===================================================================
# Configuration loaders
# ===================================================================


def load_yaml_config(path: str | Path) -> dict:
    """Load and parse a YAML configuration file.

    Args:
        path: Absolute or relative path to the YAML file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is malformed.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_city_config(city_name: str | None = None) -> dict:
    """Load city definitions from config/cities.yaml.

    Args:
        city_name: Optional city key to return a single city config.
                   If None, returns all cities.

    Returns:
        City configuration dictionary.

    Raises:
        KeyError: If the requested city is not found.
    """
    config = load_yaml_config(CONFIG_DIR / "cities.yaml")
    cities = config.get("cities", config)

    if city_name is None:
        return cities

    # Normalize hyphens to underscores to tolerate CLI args like new-york-city
    city_key = city_name.replace("-", "_")

    if city_key not in cities:
        available = ", ".join(cities.keys())
        raise KeyError(f"City '{city_name}' not found. Available: {available}")

    return cities[city_key]


# ===================================================================
# Price parsing
# ===================================================================


def clean_price(value: str | None) -> float | None:
    """Parse a price string into a float value.

    Handles Airbnb's currency-prefixed formats: "$1,250.00", "€99.00",
    "£1,234.56". Returns None for empty, null, or unparseable values.

    Args:
        value: Raw price string from CSV.

    Returns:
        Float price, or None if unparseable.

    Examples:
        >>> clean_price("$1,250.00")
        1250.0
        >>> clean_price("€99.00")
        99.0
        >>> clean_price("")
        >>> clean_price(None)
    """
    if value is None or not isinstance(value, str):
        return None

    cleaned = CURRENCY_SYMBOL_RE.sub("", value).strip()
    if not cleaned:
        return None

    try:
        return float(cleaned)
    except ValueError:
        logger.debug("Unparseable price value: %r", value)
        return None


def detect_price_currency(value: str | None) -> str | None:
    """Detect currency symbol from a price string.

    Args:
        value: Raw price string.

    Returns:
        Currency symbol character, or None if not detected.

    Examples:
        >>> detect_price_currency("$1,250.00")
        '$'
        >>> detect_price_currency("€99.00")
        '€'
    """
    if value is None or not isinstance(value, str):
        return None

    for symbol in ("$", "€", "£"):
        if symbol in value:
            return symbol
    return None


# ===================================================================
# Boolean parsing
# ===================================================================


def parse_boolean(value: str | None) -> bool | None:
    """Convert Airbnb's 't'/'f' string encoding to Python bool.

    Args:
        value: Raw string, expected 't' or 'f'.

    Returns:
        True, False, or None for unrecognized / missing values.

    Examples:
        >>> parse_boolean("t")
        True
        >>> parse_boolean("f")
        False
        >>> parse_boolean("N/A")
    """
    if value is None or not isinstance(value, str):
        return None
    return BOOLEAN_MAP.get(value.strip().lower())


# ===================================================================
# Amenities parsing
# ===================================================================


def parse_amenities(value: str | None) -> list[str]:
    """Parse amenities from a stringified JSON array or set notation.

    Inside Airbnb encodes amenities in two historical formats:
      - JSON array:   '["Wifi", "Kitchen", "Pool"]'
      - Set notation:  '{"Wifi", "Kitchen"}'   (older scrapes)

    Args:
        value: Raw amenities string.

    Returns:
        Sorted list of unique amenity strings, empty list if unparseable.

    Examples:
        >>> parse_amenities('["Wifi", "Kitchen"]')
        ['Kitchen', 'Wifi']
        >>> parse_amenities('{"Wifi", "Kitchen"}')
        ['Kitchen', 'Wifi']
    """
    if value is None or not isinstance(value, str) or not value.strip():
        return []

    text = value.strip()

    # Attempt 1: standard JSON array
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return sorted({str(item).strip() for item in parsed if item})
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt 2: set-notation fallback  {"item1", "item2"}
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1]
        items = set()
        for part in inner.split(","):
            clean = part.strip().strip("\"'").strip()
            if clean:
                items.add(clean)
        return sorted(items)

    logger.debug("Unparseable amenities (first 80 chars): %r", text[:80])
    return []


# ===================================================================
# Bathrooms text parsing
# ===================================================================


def parse_bathrooms_text(value: str | None) -> tuple[float | None, bool]:
    """Extract numeric bathroom count and shared flag from text.

    Handles patterns like:
      "1.5 baths"          → (1.5, False)
      "Half-bath"          → (0.5, False)
      "Shared half-bath"   → (0.5, True)
      "1 private bath"     → (1.0, False)
      "0 baths"            → (0.0, False)

    Args:
        value: Raw bathrooms_text string.

    Returns:
        Tuple of (count, is_shared).
        Count is None if no numeric or keyword could be extracted.

    Examples:
        >>> parse_bathrooms_text("1.5 baths")
        (1.5, False)
        >>> parse_bathrooms_text("Shared half-bath")
        (0.5, True)
    """
    if value is None or not isinstance(value, str) or not value.strip():
        return None, False

    text = value.strip().lower()
    is_shared = "shared" in text

    # Try extracting a numeric value first
    match = NUMERIC_EXTRACT_RE.search(text)
    if match:
        return float(match.group(1)), is_shared

    # Handle keyword-only patterns
    if "half" in text:
        return 0.5, is_shared

    return None, is_shared


# ===================================================================
# Percentage rate parsing
# ===================================================================


def parse_rate_pct(value: str | None) -> float | None:
    """Parse percentage strings into proportions (0.0–1.0).

    Args:
        value: Raw string like "95%", "N/A", or "".

    Returns:
        Float proportion, or None for missing / invalid values.

    Examples:
        >>> parse_rate_pct("95%")
        0.95
        >>> parse_rate_pct("N/A")
    """
    if value is None or not isinstance(value, str):
        return None

    cleaned = value.strip().rstrip("%").strip()
    if not cleaned or cleaned.upper() == "N/A":
        return None

    try:
        return float(cleaned) / 100.0
    except ValueError:
        logger.debug("Unparseable rate value: %r", value)
        return None


# ===================================================================
# Host verifications parsing
# ===================================================================


def parse_host_verifications(value: str | None) -> list[str]:
    """Parse host verifications from a stringified Python list.

    Args:
        value: Raw string like "['email', 'phone', 'work_email']".

    Returns:
        List of verification method strings.

    Examples:
        >>> parse_host_verifications("['email', 'phone']")
        ['email', 'phone']
    """
    if value is None or not isinstance(value, str) or not value.strip():
        return []

    # Attempt safe Python literal parsing
    try:
        parsed = literal_eval(value.strip())
        if isinstance(parsed, (list, set, tuple)):
            return sorted(str(item).strip() for item in parsed if item)
    except (ValueError, SyntaxError):
        pass

    # Fallback: manual cleanup
    cleaned = value.strip().strip("[]{}").replace("'", "").replace('"', "")
    items = [item.strip() for item in cleaned.split(",") if item.strip()]
    return sorted(items)


# ===================================================================
# Text cleaning
# ===================================================================


def strip_html(text: str | None) -> str | None:
    """Remove HTML tags and decode common entities from text fields.

    Args:
        text: Raw text potentially containing HTML markup.

    Returns:
        Cleaned text string, or None.

    Examples:
        >>> strip_html("<b>Hello</b> world &amp; more")
        'Hello world & more'
    """
    if text is None or not isinstance(text, str):
        return None

    result = HTML_TAG_RE.sub("", text)
    # Decode common HTML entities
    result = (
        result.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("<br>", "\n")
        .replace("<br/>", "\n")
    )
    return result.strip() or None


# ===================================================================
# Schema utilities
# ===================================================================


def compute_schema_hash(columns: list[str]) -> str:
    """Compute a deterministic hash of column names for drift detection.

    Columns are sorted before hashing, so column ordering changes
    do not produce a different hash — only additions/removals do.

    Args:
        columns: List of column name strings.

    Returns:
        MD5 hex digest string.

    Examples:
        >>> compute_schema_hash(["id", "name", "price"])
        'a3c9...'  # deterministic 32-char hex
    """
    signature = "|".join(sorted(columns))
    return hashlib.md5(signature.encode("utf-8")).hexdigest()


# ===================================================================
# File type inference
# ===================================================================

# Mapping of filename stems to logical file types
_FILE_TYPE_MAP: dict[str, str] = {
    "listings": "listings",
    "calendar": "calendar",
    "reviews": "reviews",
    "neighbourhoods": "neighbourhoods",
    "neighborhoods": "neighbourhoods",
}


def infer_file_type(filepath: str | Path) -> str:
    """Infer the logical file type from a filename.

    Args:
        filepath: Path to the data file.

    Returns:
        One of: 'listings', 'calendar', 'reviews', 'neighbourhoods', 'unknown'.

    Examples:
        >>> infer_file_type("data/raw/paris/listings.csv.gz")
        'listings'
        >>> infer_file_type("data/raw/paris/calendar.csv.gz")
        'calendar'
    """
    stem = Path(filepath).stem
    # Handle double extensions like .csv.gz
    if stem.endswith(".csv"):
        stem = stem[: -len(".csv")]

    return _FILE_TYPE_MAP.get(stem.lower(), "unknown")


# ===================================================================
# Directory helpers
# ===================================================================


def ensure_dirs(*dirs: Path) -> None:
    """Create directories if they don't exist.

    Args:
        *dirs: One or more Path objects to create.
    """
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory: %s", d)


def get_canonical_city_name(city_name: str) -> str:
    """Get the canonical city name (city_slug) from config."""
    try:
        config = load_city_config(city_name)
        return config.get("city_slug", city_name)
    except KeyError:
        return city_name


def get_raw_data_dir(city_name: str) -> Path:
    """Get the raw data directory path for a city.

    Args:
        city_name: City key from cities.yaml.

    Returns:
        Path object: data/raw/{canonical_city_name}/
    """
    return DATA_DIR / "raw" / get_canonical_city_name(city_name)


def get_output_dir(subdir: str) -> Path:
    """Get an output subdirectory path, creating it if needed.

    Args:
        subdir: Subdirectory name (e.g., 'schemas', 'profiles', 'quality').

    Returns:
        Path object: outputs/{subdir}/
    """
    path = OUTPUT_DIR / subdir
    ensure_dirs(path)
    return path


def get_staging_dir(city_name: str) -> Path:
    """Get the staging (cleaned) data directory for a city.

    Args:
        city_name: City key from cities.yaml.

    Returns:
        Path object: data/staging/{canonical_city_name}/
    """
    path = DATA_DIR / "staging" / get_canonical_city_name(city_name)
    ensure_dirs(path)
    return path


def get_rejected_dir(city_name: str) -> Path:
    """Get the rejected records directory for a city.

    Records that fail validation during cleaning are written here
    with their failure flags for audit.

    Args:
        city_name: City key from cities.yaml.

    Returns:
        Path object: data/staging/{canonical_city_name}/_rejected/
    """
    path = DATA_DIR / "staging" / get_canonical_city_name(city_name) / "_rejected"
    ensure_dirs(path)
    return path


def get_enriched_dir() -> Path:
    """Get the enriched (gold-layer) data directory.

    Returns:
        Path object: data/enriched/
    """
    path = DATA_DIR / "enriched"
    ensure_dirs(path)
    return path


def get_db_path() -> Path:
    """Get the DuckDB database file path.

    Returns:
        Path object: data/airbnb.duckdb
    """
    ensure_dirs(DATA_DIR)
    return DATA_DIR / "airbnb.duckdb"


def filter_raw_files(raw_dir: Path) -> list[Path]:
    """Discover data files and deduplicate .csv vs .csv.gz.
    
    If both exist, prefer the .csv.gz file to avoid processing the
    same dataset twice which can cause memory pressure.
    """
    if not raw_dir.exists():
        return []
        
    files = {}
    all_files = sorted(list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.csv.gz")))
    for p in all_files:
        base_name = p.name.replace(".csv.gz", "").replace(".csv", "")
        if base_name not in files:
            files[base_name] = p
        elif p.name.endswith(".csv.gz") and files[base_name].name.endswith(".csv"):
            files[base_name] = p
            
    return sorted(list(files.values()))
