"""Feature engineering pipeline for ML models.

Builds a reproducible feature matrix from the DuckDB star schema.
All transformations are pure functions operating on DataFrames,
making them testable in isolation without a database connection.

The main entry point is ``build_feature_matrix()``, which:
  1. Queries the star schema for raw listing attributes
  2. Extracts amenity flags from JSON strings
  3. Computes distance-to-city-centre
  4. Creates interaction terms
  5. Handles missing values (imputation + indicator columns)
  6. Encodes categoricals (one-hot with cardinality cap)

Usage (from CLI via orchestrator):
    from src.platform.feature_engineering.feature_store import build_feature_matrix
    feature_set = build_feature_matrix(config)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml

from src.platform.common.utils import get_db_path

logger = logging.getLogger(__name__)


# Data Classes
@dataclass
class FeatureSet:
    """Immutable container for a feature matrix and its metadata."""

    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]
    listing_ids: pd.Series
    metadata_columns: pd.DataFrame
    config_snapshot: dict
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


@dataclass
class TrainTestSplit:
    """Container for stratified train/test split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame
    train_indices: np.ndarray
    test_indices: np.ndarray


# Configuration Loading
def load_ml_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the ML pipeline configuration.

    Args:
        config_path: Path to ml_config.yaml. Defaults to config/ml_config.yaml.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If required keys are missing.
    """
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "config"
            / "ml_config.yaml"
        )
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"ML config not found: {config_path}")

    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    """Check that required configuration sections are present."""
    required_sections = ["target", "split", "features", "models", "evaluation"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(f"ML config missing required sections: {missing}")

    if "column" not in config["target"]:
        raise ValueError("ML config target must specify 'column'")

    features = config["features"]
    for key in ("numeric", "categorical", "boolean", "exclude"):
        if key not in features:
            raise ValueError(f"ML config features must specify '{key}'")


# Data Loading
_FEATURE_QUERY = """
    SELECT
        f.listing_id,
        c.display_name      AS city,
        c.city_name          AS city_key,
        p.room_type,
        p.property_type,
        p.accommodates,
        p.bedrooms,
        p.beds,
        p.bathrooms,
        p.amenities,
        p.amenity_count,
        p.instant_bookable,
        n.neighbourhood_group,
        h.host_is_superhost,
        h.host_identity_verified,
        h.host_response_time,
        h.host_response_rate,
        h.host_listings_count,
        h.is_professional_host,
        f.price_usd,
        f.price_local,
        f.latitude,
        f.longitude,
        f.minimum_nights,
        f.maximum_nights,
        f.number_of_reviews,
        f.number_of_reviews_ltm,
        f.review_scores_rating,
        f.review_scores_accuracy,
        f.review_scores_cleanliness,
        f.review_scores_checkin,
        f.review_scores_communication,
        f.review_scores_location,
        f.review_scores_value,
        f.reviews_per_month,
        f.availability_30,
        f.availability_60,
        f.availability_90,
        f.availability_365,
        f.occupancy_rate_pct,
        f.host_tenure_years
    FROM fact_listing_snapshot f
    JOIN dim_city          c ON f.city_key          = c.city_key
    JOIN dim_property      p ON f.property_key      = p.property_key
    JOIN dim_neighbourhood n ON f.neighbourhood_key = n.neighbourhood_key
    JOIN dim_host          h ON f.host_key          = h.host_key
    WHERE f.price_usd IS NOT NULL
      AND f.price_usd > 0
"""


def load_raw_data(db_path: str | Path | None = None) -> pd.DataFrame:
    """Query the star schema and return a raw DataFrame for feature engineering.

    Args:
        db_path: Path to the DuckDB file. Defaults to data/airbnb.duckdb.

    Returns:
        DataFrame with all columns needed for feature engineering.
    """
    if db_path is None:
        db_path = get_db_path()

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"DuckDB file not found: {db_path}. Run the data pipeline first."
        )

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(_FEATURE_QUERY).fetchdf()
    finally:
        con.close()

    logger.info("Loaded %d listings from star schema", len(df))
    return df


# Amenity Flag Extraction
def extract_amenity_flags(
    amenities_series: pd.Series,
    keyword_map: dict[str, list[str]],
) -> pd.DataFrame:
    """Parse JSON/string amenity lists into binary flag columns.

    Each keyword group (e.g., ``has_wifi: ["wifi", "wi-fi"]``) becomes
    a binary column. Matching is case-insensitive.

    Args:
        amenities_series: Series of JSON-encoded amenity strings.
        keyword_map: {flag_name: [keywords]} from config.

    Returns:
        DataFrame with one boolean column per flag.
    """
    results = {
        flag: np.zeros(len(amenities_series), dtype=np.int8) for flag in keyword_map
    }

    for idx, raw in enumerate(amenities_series):
        if pd.isna(raw) or not raw:
            continue

        # Normalise: try JSON parse, fall back to string matching
        try:
            amenities_text = " ".join(json.loads(raw)).lower()
        except (json.JSONDecodeError, TypeError):
            amenities_text = str(raw).lower()

        for flag, keywords in keyword_map.items():
            if any(kw in amenities_text for kw in keywords):
                results[flag][idx] = 1

    df = pd.DataFrame(results, index=amenities_series.index)
    logger.info(
        "Extracted %d amenity flags (%d listings)",
        len(keyword_map),
        len(amenities_series),
    )
    return df


# Distance Features
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two points in kilometres."""
    R = 6371.0
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def compute_distance_to_centre(
    df: pd.DataFrame,
    city_centres: dict[str, dict[str, float]],
    city_column: str = "city_key",
) -> pd.Series:
    """Compute Haversine distance from each listing to its city centre.

    Args:
        df: DataFrame with latitude, longitude, and city_column.
        city_centres: {city_key: {lat, lon}} from config.
        city_column: Column identifying the city for each listing.

    Returns:
        Series of distances in kilometres.
    """
    distances = np.full(len(df), np.nan)

    for city_key, centre in city_centres.items():
        mask = df[city_column] == city_key
        if not mask.any():
            continue

        for idx in df.index[mask]:
            lat = df.at[idx, "latitude"]
            lon = df.at[idx, "longitude"]
            if pd.notna(lat) and pd.notna(lon):
                distances[df.index.get_loc(idx)] = _haversine_km(
                    lat, lon, centre["lat"], centre["lon"]
                )

    return pd.Series(distances, index=df.index, name="distance_to_centre_km")


def compute_distance_to_centre_vectorised(
    df: pd.DataFrame,
    city_centres: dict[str, dict[str, float]],
    city_column: str = "city_key",
) -> pd.Series:
    """Vectorised Haversine distance — much faster for large datasets."""
    R = 6371.0
    distances = pd.Series(np.nan, index=df.index, name="distance_to_centre_km")

    for city_key, centre in city_centres.items():
        mask = df[city_column] == city_key
        if not mask.any():
            continue

        lat = np.radians(df.loc[mask, "latitude"].values.astype(float))
        lon = np.radians(df.loc[mask, "longitude"].values.astype(float))
        c_lat = math.radians(centre["lat"])
        c_lon = math.radians(centre["lon"])

        dlat = lat - c_lat
        dlon = lon - c_lon
        a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(c_lat) * np.sin(dlon / 2) ** 2
        d = R * 2 * np.arcsin(np.sqrt(a))

        distances.loc[mask] = d

    return distances


# Interaction Terms
def create_interaction_terms(
    df: pd.DataFrame,
    interactions: list[list[str]],
) -> pd.DataFrame:
    """Create multiplicative interaction features.

    Args:
        df: Feature DataFrame (numeric columns expected).
        interactions: List of [feature_a, feature_b] pairs.

    Returns:
        DataFrame with interaction columns named ``{a}_x_{b}``.
    """
    result = pd.DataFrame(index=df.index)

    for pair in interactions:
        if len(pair) != 2:
            logger.warning("Skipping malformed interaction pair: %s", pair)
            continue
        col_a, col_b = pair
        if col_a not in df.columns or col_b not in df.columns:
            logger.warning("Interaction columns missing: %s × %s", col_a, col_b)
            continue

        name = f"{col_a}_x_{col_b}"
        result[name] = df[col_a].fillna(0) * df[col_b].fillna(0)

    return result


# Missing Value Handling
def handle_missing_values(
    df: pd.DataFrame,
    config: dict,
    numeric_cols: list[str],
    boolean_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Impute missing values according to the configured strategy.

    Args:
        df: Input DataFrame.
        config: ``missing_values`` section from ml_config.yaml.
        numeric_cols: Columns to impute with median.
        boolean_cols: Columns to impute with False.
        categorical_cols: Columns to impute with placeholder string.

    Returns:
        (imputed_df, indicator_columns) — the DataFrame and list of
        ``is_missing_*`` indicator columns that were added.
    """
    result = df.copy()
    indicators: list[str] = []

    add_indicators = config.get("add_indicators", False)

    # Numeric imputation
    for col in numeric_cols:
        if col not in result.columns:
            continue
        n_missing = result[col].isna().sum()
        if n_missing > 0:
            if add_indicators:
                ind_col = f"is_missing_{col}"
                result[ind_col] = result[col].isna().astype(np.int8)
                indicators.append(ind_col)
            result[col] = result[col].fillna(result[col].median())

    # Boolean imputation
    for col in boolean_cols:
        if col not in result.columns:
            continue
        result[col] = result[col].fillna(False).astype(int)

    # Categorical imputation
    placeholder = config.get("categorical", "__MISSING__")
    for col in categorical_cols:
        if col not in result.columns:
            continue
        result[col] = result[col].fillna(placeholder).astype(str)

    return result, indicators


# Categorical Encoding
def encode_categoricals(
    df: pd.DataFrame,
    columns: list[str],
    max_cardinality: int = 20,
) -> pd.DataFrame:
    """One-hot encode categorical columns with a cardinality cap.

    Categories with fewer than ``n / max_cardinality`` occurrences
    are collapsed into ``__OTHER__`` to prevent high-dimensional sparse
    features.

    Args:
        df: Input DataFrame.
        columns: Categorical columns to encode.
        max_cardinality: Maximum number of distinct categories per column.

    Returns:
        DataFrame with original columns replaced by one-hot dummies.
    """
    result = df.copy()
    cols_to_drop = []

    for col in columns:
        if col not in result.columns:
            continue

        series = result[col].astype(str)

        # Cap cardinality
        top_cats = series.value_counts().nlargest(max_cardinality).index.tolist()
        series = series.where(series.isin(top_cats), "__OTHER__")

        dummies = pd.get_dummies(series, prefix=col, drop_first=True, dtype=np.int8)
        result = pd.concat([result, dummies], axis=1)
        cols_to_drop.append(col)

    result = result.drop(columns=cols_to_drop, errors="ignore")
    return result


# Winsorisation
def winsorise(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip values at the given quantile bounds.

    Args:
        series: Numeric series to clip.
        lower: Lower quantile (default: 1st percentile).
        upper: Upper quantile (default: 99th percentile).

    Returns:
        Clipped series.
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


# Main Entry Point
def build_feature_matrix(
    config: dict[str, Any],
    db_path: str | Path | None = None,
) -> FeatureSet:
    """Build a complete feature matrix from the DuckDB star schema.

    This is the main entry point for feature engineering. It:
      1. Loads raw data from DuckDB
      2. Filters to valid price records and winsorises target
      3. Extracts amenity flags
      4. Computes distance to city centre
      5. Creates interaction terms
      6. Handles missing values
      7. Encodes categoricals

    Args:
        config: Parsed ml_config.yaml dictionary.
        db_path: Optional override for the DuckDB path.

    Returns:
        FeatureSet with feature matrix X, target y, and metadata.
    """
    logger.info("Building feature matrix...")

    # Load raw data
    raw_df = load_raw_data(db_path)
    target_col = config["target"]["column"]

    # Winsorise target (remove extreme outliers)
    raw_df = raw_df[raw_df[target_col] > 0].copy()
    p99 = raw_df[target_col].quantile(0.99)
    raw_df = raw_df[raw_df[target_col] <= p99].copy()
    raw_df = raw_df.reset_index(drop=True)

    logger.info("After winsorisation: %d listings", len(raw_df))

    # Extract target
    y_raw = raw_df[target_col].copy()
    transform = config["target"].get("transform", "log1p")
    if transform == "log1p":
        y = np.log1p(y_raw)
    else:
        y = y_raw.copy()
    y.name = "target"

    # Preserve metadata columns for stratified evaluation
    meta_cols = [
        "city",
        "city_key",
        "room_type",
        "neighbourhood_group",
        "is_professional_host",
        "host_is_superhost",
    ]
    meta_cols = [c for c in meta_cols if c in raw_df.columns]
    metadata = raw_df[meta_cols].copy()
    metadata["price_quintile"] = pd.qcut(
        y_raw, q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop"
    )
    metadata["has_reviews"] = (raw_df["number_of_reviews"] > 0).astype(int)

    listing_ids = raw_df["listing_id"].copy()

    feat_cfg = config["features"]

    # Amenity flags
    amenity_flags_df = pd.DataFrame(index=raw_df.index)
    amenity_map = feat_cfg.get("amenity_flags", {})
    if amenity_map and "amenities" in raw_df.columns:
        amenity_flags_df = extract_amenity_flags(raw_df["amenities"], amenity_map)

    # Distance to centre
    distance_series = pd.Series(
        np.nan, index=raw_df.index, name="distance_to_centre_km"
    )
    city_centres = feat_cfg.get("city_centres", {})
    if city_centres:
        distance_series = compute_distance_to_centre_vectorised(
            raw_df, city_centres, city_column="city_key"
        )

    # Start assembling feature DataFrame
    numeric_cols = [c for c in feat_cfg.get("numeric", []) if c in raw_df.columns]
    boolean_cols = [c for c in feat_cfg.get("boolean", []) if c in raw_df.columns]
    categorical_cols = [
        c for c in feat_cfg.get("categorical", []) if c in raw_df.columns
    ]

    # Gather numeric + boolean columns from raw
    feature_df = raw_df[numeric_cols + boolean_cols].copy()

    # Add amenity flags
    feature_df = pd.concat([feature_df, amenity_flags_df], axis=1)

    # Add distance
    feature_df["distance_to_centre_km"] = distance_series

    # Add categorical columns (will be encoded later)
    for col in categorical_cols:
        feature_df[col] = raw_df[col]

    # Interaction terms
    interactions = feat_cfg.get("interactions", [])
    if interactions:
        interaction_df = create_interaction_terms(feature_df, interactions)
        feature_df = pd.concat([feature_df, interaction_df], axis=1)

    # Handle missing values
    mv_config = config.get("missing_values", {})
    all_numeric = (
        numeric_cols + list(amenity_flags_df.columns) + ["distance_to_centre_km"]
    )
    all_numeric += [c for c in feature_df.columns if "_x_" in c]

    feature_df, indicator_cols = handle_missing_values(
        feature_df, mv_config, all_numeric, boolean_cols, categorical_cols
    )

    # Encode categoricals
    max_card = feat_cfg.get("max_cardinality", 20)
    feature_df = encode_categoricals(
        feature_df, categorical_cols, max_cardinality=max_card
    )

    # Remove any excluded columns that leaked through
    exclude = set(feat_cfg.get("exclude", []))
    feature_df = feature_df.drop(
        columns=[c for c in feature_df.columns if c in exclude], errors="ignore"
    )

    # Final: ensure all columns are numeric and catch any missed NaNs
    for col in feature_df.columns:
        if feature_df[col].dtype == object:
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
        if feature_df[col].isna().any():
            feature_df[col] = feature_df[col].fillna(0)

    feature_names = list(feature_df.columns)

    logger.info(
        "Feature matrix built: %d samples × %d features",
        len(feature_df),
        len(feature_names),
    )

    return FeatureSet(
        X=feature_df,
        y=y,
        feature_names=feature_names,
        listing_ids=listing_ids,
        metadata_columns=metadata,
        config_snapshot=config,
    )


# Train / Test Split
def prepare_train_test_split(
    feature_set: FeatureSet,
    config: dict[str, Any],
) -> TrainTestSplit:
    """Create a stratified train/test split.

    Stratification preserves the distribution of city × price_quintile
    across both splits, ensuring representative evaluation.

    Args:
        feature_set: The FeatureSet from build_feature_matrix().
        config: Parsed ml_config.yaml dictionary.

    Returns:
        TrainTestSplit with train/test DataFrames and indices.
    """
    from sklearn.model_selection import train_test_split

    split_cfg = config["split"]
    test_size = split_cfg.get("test_size", 0.20)
    random_state = split_cfg.get("random_state", 42)

    # Build stratification groups
    meta = feature_set.metadata_columns
    strat_cols = split_cfg.get("stratify_by", ["city", "price_quintile"])
    available_strat = [c for c in strat_cols if c in meta.columns]

    if available_strat:
        strat_group = meta[available_strat].astype(str).agg("_".join, axis=1)
        # Collapse rare groups (< 2 members) to avoid split failure
        counts = strat_group.value_counts()
        rare = counts[counts < 2].index
        strat_group = strat_group.replace(rare, "__RARE__")
    else:
        strat_group = None

    indices = np.arange(len(feature_set.X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_group,
    )

    return TrainTestSplit(
        X_train=feature_set.X.iloc[train_idx].reset_index(drop=True),
        X_test=feature_set.X.iloc[test_idx].reset_index(drop=True),
        y_train=feature_set.y.iloc[train_idx].reset_index(drop=True),
        y_test=feature_set.y.iloc[test_idx].reset_index(drop=True),
        meta_train=feature_set.metadata_columns.iloc[train_idx].reset_index(drop=True),
        meta_test=feature_set.metadata_columns.iloc[test_idx].reset_index(drop=True),
        train_indices=train_idx,
        test_indices=test_idx,
    )
