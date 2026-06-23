"""Data cleaning and standardization pipeline (Section 3.2).

Transforms raw CSV/GZ files from the landing zone into clean, typed,
validated Parquet files in the staging zone. Each file type has a
dedicated cleaning function that applies:

  1. Type coercion     — price→float, boolean→bool, date→Date, %→proportion
  2. Special parsing   — bathrooms_text→float, amenities count, host verifications
  3. Text normalization — lowercase/strip categorical fields
  4. Missing values    — per-column strategy from config (reject/sentinel/impute/null)
  5. Validation flags  — _is_valid + _validation_flags columns
  6. Partitioning      — valid records → staging, invalid → _rejected/

ALL transformations use vectorized Polars expressions. No row-level
Python loops (apply/map_elements) are used — ensuring sub-second
performance even on calendar files with 24M+ rows.

Outputs:
  - data/staging/{city}/{file_type}.parquet       (clean records)
  - data/staging/{city}/_rejected/{file_type}.parquet  (failed records + flags)

Usage:
    from src.platform.data_engineering.ingestion.cleaner import clean_city
    results = clean_city("paris")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.platform.common.utils import (
    CONFIG_DIR,
    get_output_dir,
    get_raw_data_dir,
    get_rejected_dir,
    get_staging_dir,
    load_yaml_config,
)

logger = logging.getLogger(__name__)

# Schema inference rows for Polars CSV reader
_SCHEMA_INFERENCE_ROWS = 10_000


# ===================================================================
# Result data class
# ===================================================================


@dataclass
class CleaningResult:
    """Result of cleaning a single file type for a city."""

    file_type: str
    city: str
    source_file: str
    input_rows: int
    output_rows: int
    rejected_rows: int
    output_path: str
    rejected_path: str | None
    columns_cleaned: dict[str, str] = field(default_factory=dict)
    imputed_columns: list[str] = field(default_factory=list)


# ===================================================================
# Configuration loading
# ===================================================================


def _load_cleaning_config(file_type: str) -> dict[str, Any]:
    """Load cleaning rules for a specific file type.

    Args:
        file_type: Logical file type ('listings', 'calendar', etc.).

    Returns:
        Cleaning configuration dict for the file type.

    Raises:
        KeyError: If the file type is not defined in cleaning_rules.yaml.
    """
    config = load_yaml_config(CONFIG_DIR / "cleaning_rules.yaml")

    if file_type not in config:
        raise KeyError(
            f"No cleaning rules defined for file type '{file_type}'. "
            f"Available: {list(config.keys())}"
        )

    return config[file_type]


# ===================================================================
# Type coercion — vectorized Polars expressions
# ===================================================================


def _clean_price_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Strip currency symbols and cast price strings to Float64.

    "$1,250.00" → 1250.0,  "€99.00" → 99.0,  "" → null

    Args:
        df: Input DataFrame.
        columns: Price column names to clean.

    Returns:
        DataFrame with price columns cast to Float64.
    """
    exprs = []
    for col in columns:
        if col not in df.columns:
            continue
        exprs.append(
            pl.col(col)
            .str.replace_all(r"[\$€£,]", "")
            .str.strip_chars()
            .cast(pl.Float64, strict=False)
            .alias(col)
        )

    return df.with_columns(exprs) if exprs else df


def _cast_boolean_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Cast Airbnb's 't'/'f' string encoding to native Boolean.

    "t" → True,  "f" → False,  anything else → null

    Args:
        df: Input DataFrame.
        columns: Boolean column names to cast.

    Returns:
        DataFrame with boolean columns cast to Boolean type.
    """
    exprs = []
    for col in columns:
        if col not in df.columns:
            continue
        exprs.append(
            pl.when(pl.col(col) == "t")
            .then(True)
            .when(pl.col(col) == "f")
            .then(False)
            .otherwise(None)
            .alias(col)
        )

    return df.with_columns(exprs) if exprs else df


def _parse_date_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Parse date strings (YYYY-MM-DD) to native Date type.

    Args:
        df: Input DataFrame.
        columns: Date column names to parse.

    Returns:
        DataFrame with date columns cast to Date type.
    """
    exprs = []
    for col in columns:
        if col not in df.columns:
            continue
        # Only parse if column is currently string type
        if df[col].dtype in (pl.Utf8, pl.String):
            exprs.append(pl.col(col).str.to_date("%Y-%m-%d", strict=False).alias(col))

    return df.with_columns(exprs) if exprs else df


def _clean_percentage_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Strip '%' sign and convert to proportion (0.0–1.0).

    "95%" → 0.95,  "N/A" → null,  "" → null

    Args:
        df: Input DataFrame.
        columns: Percentage column names to clean.

    Returns:
        DataFrame with percentage columns as Float64 proportions.
    """
    exprs = []
    for col in columns:
        if col not in df.columns:
            continue
        exprs.append(
            pl.col(col).str.replace("%", "").str.strip_chars().cast(pl.Float64, strict=False)
            / 100.0
        )
        # Polars requires explicit alias after arithmetic
        exprs[-1] = exprs[-1].alias(col)

    return df.with_columns(exprs) if exprs else df


# ===================================================================
# Special column parsing — vectorized
# ===================================================================


def _parse_bathrooms_column(df: pl.DataFrame) -> pl.DataFrame:
    """Parse bathrooms_text into numeric bathrooms + shared flag.

    "1.5 baths"        → bathrooms=1.5, bathrooms_shared=False
    "Shared half-bath"  → bathrooms=0.5, bathrooms_shared=True
    "Half-bath"         → bathrooms=0.5, bathrooms_shared=False

    Uses Polars string expressions (no row-level Python).

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with 'bathrooms' (Float64) and 'bathrooms_shared' (Boolean)
        columns added. Original 'bathrooms_text' is preserved.
    """
    if "bathrooms_text" not in df.columns:
        return df

    lowered = pl.col("bathrooms_text").str.to_lowercase()

    # Extract numeric value from text
    numeric_expr = (
        pl.col("bathrooms_text")
        .str.extract(r"(\d+\.?\d*)", group_index=1)
        .cast(pl.Float64, strict=False)
    )

    # Handle "half-bath" patterns without numeric prefix
    bathrooms_expr = (
        pl.when(numeric_expr.is_not_null())
        .then(numeric_expr)
        .when(lowered.str.contains("half"))
        .then(pl.lit(0.5))
        .otherwise(None)
        .alias("bathrooms")
    )

    # Detect shared bathrooms
    shared_expr = (
        pl.when(pl.col("bathrooms_text").is_not_null())
        .then(lowered.str.contains("shared"))
        .otherwise(None)
        .alias("bathrooms_shared")
    )

    return df.with_columns([bathrooms_expr, shared_expr])


def _count_amenities(df: pl.DataFrame) -> pl.DataFrame:
    """Add amenity_count column from the amenities JSON string.

    Counts elements by counting delimiters in the JSON array string.
    Full amenity parsing is deferred to the enrichment stage.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with 'amenity_count' (Int32) column added.
    """
    if "amenities" not in df.columns:
        return df

    return df.with_columns(
        pl.when(
            pl.col("amenities").is_not_null()
            & (pl.col("amenities").str.len_chars() > 2)  # more than "[]"
        )
        .then(pl.col("amenities").str.count_matches(r'","').cast(pl.Int32) + 1)
        .otherwise(pl.lit(0).cast(pl.Int32))
        .alias("amenity_count")
    )


def _parse_host_verifications_column(df: pl.DataFrame) -> pl.DataFrame:
    """Add verification_count from host_verifications string.

    Full list parsing is deferred; this just counts elements.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with 'host_verification_count' (Int32) column added.
    """
    if "host_verifications" not in df.columns:
        return df

    return df.with_columns(
        pl.when(
            pl.col("host_verifications").is_not_null()
            & (pl.col("host_verifications").str.len_chars() > 2)
        )
        .then(pl.col("host_verifications").str.count_matches(r"'[^']+'").cast(pl.Int32))
        .otherwise(pl.lit(0).cast(pl.Int32))
        .alias("host_verification_count")
    )


def _strip_html_column(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Remove HTML tags from a text column using Polars expressions.

    Args:
        df: Input DataFrame.
        col: Column name to clean.

    Returns:
        DataFrame with HTML tags stripped from the column.
    """
    if col not in df.columns:
        return df

    return df.with_columns(
        pl.col(col)
        .str.replace_all(r"<[^>]+>", "")
        .str.replace_all(r"&amp;", "&")
        .str.replace_all(r"&lt;", "<")
        .str.replace_all(r"&gt;", ">")
        .str.replace_all(r"&quot;", '"')
        .str.strip_chars()
        .alias(col)
    )


def _apply_special_columns(
    df: pl.DataFrame,
    special_config: dict[str, str],
) -> pl.DataFrame:
    """Apply special column parsing based on config.

    Args:
        df: Input DataFrame.
        special_config: Mapping of column_name → parsing_method.

    Returns:
        DataFrame with special columns parsed.
    """
    for col_name, method in special_config.items():
        if col_name not in df.columns:
            continue

        if method == "parse_bathrooms":
            df = _parse_bathrooms_column(df)
        elif method == "parse_amenities":
            df = _count_amenities(df)
        elif method == "parse_list":
            df = _parse_host_verifications_column(df)
        elif method == "strip_html":
            df = _strip_html_column(df, col_name)
        else:
            logger.warning("Unknown special parsing method: %s for column %s", method, col_name)

    return df


# ===================================================================
# Text normalization
# ===================================================================


def _normalize_text_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Normalize text columns: strip whitespace, standardize casing.

    Does NOT force lowercase on categorical values like property_type
    or room_type — preserves Airbnb's original casing for downstream
    enum matching. Only strips leading/trailing whitespace and collapses
    internal multi-spaces.

    Args:
        df: Input DataFrame.
        columns: Column names to normalize.

    Returns:
        DataFrame with normalized text columns.
    """
    exprs = []
    for col in columns:
        if col not in df.columns:
            continue
        exprs.append(
            pl.col(col)
            .str.strip_chars()
            .str.replace_all(r"\s{2,}", " ")  # collapse multi-spaces
            .alias(col)
        )

    return df.with_columns(exprs) if exprs else df


# ===================================================================
# Missing value strategies
# ===================================================================


def _apply_missing_strategies(
    df: pl.DataFrame,
    strategies: dict[str, Any],
) -> tuple[pl.DataFrame, list[str]]:
    """Apply per-column missing value strategies from config.

    Strategies:
      - reject:               Leave null; validation will flag and reject the record
      - explicit_null:         Leave null intentionally (no action needed)
      - sentinel:              Fill with a specified value
      - impute_median_by_group: Fill with group median; add _imputed flag column
      - impute_zero:           Fill with 0

    Args:
        df: Input DataFrame.
        strategies: Dict of column_name → strategy spec from cleaning_rules.yaml.

    Returns:
        Tuple of (cleaned DataFrame, list of imputed column names).
    """
    imputed_columns: list[str] = []

    for col_name, spec in strategies.items():
        if col_name not in df.columns:
            continue

        if isinstance(spec, dict):
            strategy = spec.get("strategy", "explicit_null")
        else:
            strategy = str(spec)

        if strategy in ("reject", "explicit_null"):
            # No action — nulls stay; validation rules handle rejection
            continue

        elif strategy == "sentinel":
            fill_value = spec.get("fill_value", "Unknown")
            df = df.with_columns(pl.col(col_name).fill_null(pl.lit(fill_value)).alias(col_name))
            logger.debug("Sentinel fill: %s → '%s'", col_name, fill_value)

        elif strategy == "impute_zero":
            df = df.with_columns(pl.col(col_name).fill_null(0).alias(col_name))

        elif strategy == "impute_median_by_group":
            group_cols = spec.get("group_columns", [])
            available_group_cols = [c for c in group_cols if c in df.columns]

            if not available_group_cols:
                # Fallback to global median if group columns unavailable
                median_val = df[col_name].median()
                if median_val is not None:
                    # Add imputed flag BEFORE filling
                    df = df.with_columns(pl.col(col_name).is_null().alias(f"_{col_name}_imputed"))
                    df = df.with_columns(
                        pl.col(col_name).fill_null(pl.lit(median_val)).alias(col_name)
                    )
                    imputed_columns.append(col_name)
                continue

            # Compute group medians from non-null rows
            median_alias = f"_median_{col_name}"
            group_medians = (
                df.filter(pl.col(col_name).is_not_null())
                .group_by(available_group_cols)
                .agg(pl.col(col_name).median().alias(median_alias))
            )

            # Mark which rows will be imputed (BEFORE filling)
            df = df.with_columns(pl.col(col_name).is_null().alias(f"_{col_name}_imputed"))

            # Join group medians and fill nulls
            df = df.join(group_medians, on=available_group_cols, how="left")
            df = df.with_columns(
                pl.when(pl.col(col_name).is_null())
                .then(pl.col(median_alias))
                .otherwise(pl.col(col_name))
                .alias(col_name)
            )
            df = df.drop(median_alias)

            imputed_columns.append(col_name)
            null_remaining = df[col_name].null_count()
            logger.info(
                "Imputed %s by group median (%s); %d nulls remaining",
                col_name,
                available_group_cols,
                null_remaining,
            )

        else:
            logger.warning("Unknown missing value strategy: %s for %s", strategy, col_name)

    return df, imputed_columns


# ===================================================================
# Post-cleaning validation flags
# ===================================================================


def _compute_validation_flags(
    df: pl.DataFrame,
    rules_config: dict[str, list[dict[str, Any]]],
) -> pl.DataFrame:
    """Compute per-record validation flags from post-cleaning rules.

    Adds two columns:
      - _is_valid (Boolean): True if all rules pass
      - _validation_flags (String): pipe-delimited list of failed rule names

    Args:
        df: Cleaned DataFrame.
        rules_config: Dict of column_name → list of rule specs from config.

    Returns:
        DataFrame with _is_valid and _validation_flags columns added.
    """
    flag_exprs: list[pl.Expr] = []

    for col_name, rules in rules_config.items():
        if col_name not in df.columns:
            continue

        for rule in rules:
            check = rule["check"]
            flag_id = f"{col_name}_{check}"

            if check == "not_null":
                flag_exprs.append(
                    pl.when(pl.col(col_name).is_null())
                    .then(pl.lit(flag_id))
                    .otherwise(pl.lit(""))
                    .alias(f"__vf_{flag_id}")
                )

            elif check == "positive":
                if df[col_name].dtype.is_numeric():
                    flag_exprs.append(
                        pl.when(pl.col(col_name).is_not_null() & (pl.col(col_name) <= 0))
                        .then(pl.lit(flag_id))
                        .otherwise(pl.lit(""))
                        .alias(f"__vf_{flag_id}")
                    )

            elif check == "range":
                if df[col_name].dtype.is_numeric():
                    min_val = rule.get("min")
                    max_val = rule.get("max")
                    conditions: list[pl.Expr] = []
                    if min_val is not None:
                        conditions.append(pl.col(col_name) < min_val)
                    if max_val is not None:
                        conditions.append(pl.col(col_name) > max_val)

                    if conditions:
                        combined = conditions[0]
                        for cond in conditions[1:]:
                            combined = combined | cond

                        flag_exprs.append(
                            pl.when(pl.col(col_name).is_not_null() & combined)
                            .then(pl.lit(flag_id))
                            .otherwise(pl.lit(""))
                            .alias(f"__vf_{flag_id}")
                        )

            elif check == "enum":
                allowed = rule.get("values", [])
                if allowed:
                    flag_exprs.append(
                        pl.when(pl.col(col_name).is_not_null() & ~pl.col(col_name).is_in(allowed))
                        .then(pl.lit(flag_id))
                        .otherwise(pl.lit(""))
                        .alias(f"__vf_{flag_id}")
                    )

    # If no rules defined, mark all records as valid
    if not flag_exprs:
        return df.with_columns(
            pl.lit(True).alias("_is_valid"),
            pl.lit("").alias("_validation_flags"),
        )

    # Apply all flag expressions in one pass
    df = df.with_columns(flag_exprs)

    # Combine individual flag columns into a single pipe-delimited string
    vf_cols = sorted(c for c in df.columns if c.startswith("__vf_"))

    df = df.with_columns(
        pl.concat_str(vf_cols, separator="|")
        .str.replace_all(r"\|{2,}", "|")  # collapse consecutive pipes
        .str.strip_chars("|")  # trim leading/trailing pipes
        .alias("_validation_flags")
    )
    df = df.with_columns((pl.col("_validation_flags") == "").alias("_is_valid"))

    # Drop intermediate flag columns
    df = df.drop(vf_cols)

    return df


# ===================================================================
# Partitioning and output
# ===================================================================


def _partition_and_save(
    df: pl.DataFrame,
    city_name: str,
    file_type: str,
) -> tuple[Path, Path | None, int, int]:
    """Partition records into valid/rejected and write Parquet files.

    Args:
        df: DataFrame with _is_valid column.
        city_name: City key.
        file_type: Logical file type.

    Returns:
        Tuple of (output_path, rejected_path, valid_count, rejected_count).
    """
    staging_dir = get_staging_dir(city_name)
    output_path = staging_dir / f"{file_type}.parquet"

    # Partition
    valid_df = df.filter(pl.col("_is_valid"))
    rejected_df = df.filter(~pl.col("_is_valid"))

    valid_count = valid_df.height
    rejected_count = rejected_df.height

    # Write valid records (drop internal columns)
    internal_cols = [c for c in valid_df.columns if c.startswith("_")]
    valid_df.drop(internal_cols).write_parquet(output_path)
    logger.info("Staging written: %s (%d rows)", output_path.name, valid_count)

    # Write rejected records (KEEP flags for audit)
    rejected_path: Path | None = None
    if rejected_count > 0:
        rejected_dir = get_rejected_dir(city_name)
        rejected_path = rejected_dir / f"{file_type}.parquet"
        rejected_df.write_parquet(rejected_path)
        logger.warning(
            "Rejected records: %s (%d rows) → %s",
            file_type,
            rejected_count,
            rejected_path.name,
        )

    return output_path, rejected_path, valid_count, rejected_count


# ===================================================================
# Per-file-type cleaning orchestrators
# ===================================================================


def _read_raw_file(city_name: str, file_type: str) -> tuple[Path, pl.DataFrame]:
    """Find and read the raw file for a given city and file type.

    Prefers the detailed (.csv.gz) version over the summary (.csv)
    when both exist.

    Args:
        city_name: City key.
        file_type: Logical file type.

    Returns:
        Tuple of (filepath, DataFrame).

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    raw_dir = get_raw_data_dir(city_name)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    # Prefer detailed over summary
    candidates = sorted(raw_dir.glob(f"{file_type}*"), reverse=True)
    gz_files = [f for f in candidates if f.name.endswith(".csv.gz")]
    csv_files = [
        f for f in candidates if f.name.endswith(".csv") and not f.name.endswith(".csv.gz")
    ]

    filepath = gz_files[0] if gz_files else (csv_files[0] if csv_files else None)
    if filepath is None:
        raise FileNotFoundError(f"No {file_type} file found in {raw_dir}")

    logger.info("Reading raw file: %s", filepath.name)

    df = pl.read_csv(
        filepath,
        infer_schema_length=_SCHEMA_INFERENCE_ROWS,
        try_parse_dates=False,
        null_values=["", "N/A", "NA"],
        truncate_ragged_lines=True,
    )

    logger.info("Loaded: %s (%d rows × %d cols)", filepath.name, df.height, df.width)
    return filepath, df


def clean_listings(city_name: str) -> CleaningResult:
    """Clean and standardize the listings dataset for a city.

    Transformation pipeline:
      1. Price → Float64 (strip currency symbols)
      2. Boolean columns → native Boolean (t/f → True/False)
      3. Date columns → native Date (YYYY-MM-DD)
      4. Percentage columns → Float64 proportion (0-1)
      5. Bathrooms text → numeric bathrooms + shared flag
      6. Amenities → amenity_count
      7. Host verifications → verification_count
      8. Text normalization (strip whitespace, collapse spaces)
      9. Missing value strategies (sentinel/impute/null per column)
     10. Validation flags → partition valid/rejected

    Args:
        city_name: City key from cities.yaml.

    Returns:
        CleaningResult with row counts and output paths.
    """
    config = _load_cleaning_config("listings")
    filepath, df = _read_raw_file(city_name, "listings")
    input_rows = df.height
    columns_cleaned: dict[str, str] = {}

    # 1. Price columns
    price_cols = config.get("price_columns", [])
    df = _clean_price_columns(df, price_cols)
    for c in price_cols:
        if c in df.columns:
            columns_cleaned[c] = "price→float64"

    # 2. Boolean columns
    bool_cols = config.get("boolean_columns", [])
    df = _cast_boolean_columns(df, bool_cols)
    for c in bool_cols:
        if c in df.columns:
            columns_cleaned[c] = "t/f→boolean"

    # 3. Date columns
    date_cols = config.get("date_columns", [])
    df = _parse_date_columns(df, date_cols)
    for c in date_cols:
        if c in df.columns:
            columns_cleaned[c] = "string→date"

    # 4. Percentage columns
    pct_cols = config.get("percentage_columns", [])
    df = _clean_percentage_columns(df, pct_cols)
    for c in pct_cols:
        if c in df.columns:
            columns_cleaned[c] = "pct_string→proportion"

    # 5. Special column parsing
    special = config.get("special_columns", {})
    df = _apply_special_columns(df, special)
    for c in special:
        if c in df.columns:
            columns_cleaned[c] = f"special:{special[c]}"

    # 6. Text normalization
    text_cols = config.get("text_normalize_columns", [])
    df = _normalize_text_columns(df, text_cols)
    for c in text_cols:
        if c in df.columns:
            columns_cleaned[c] = "text_normalized"

    # 7. Missing value strategies
    strategies = config.get("missing_value_strategies", {})
    df, imputed_cols = _apply_missing_strategies(df, strategies)

    # 8. Validation flags & partitioning
    validation_rules = config.get("validation_rules", {})
    df = _compute_validation_flags(df, validation_rules)

    output_path, rejected_path, valid_count, rejected_count = _partition_and_save(
        df, city_name, "listings"
    )

    logger.info(
        "Listings cleaning complete: %d input → %d valid + %d rejected",
        input_rows,
        valid_count,
        rejected_count,
    )

    return CleaningResult(
        file_type="listings",
        city=city_name,
        source_file=filepath.name,
        input_rows=input_rows,
        output_rows=valid_count,
        rejected_rows=rejected_count,
        output_path=str(output_path),
        rejected_path=str(rejected_path) if rejected_path else None,
        columns_cleaned=columns_cleaned,
        imputed_columns=imputed_cols,
    )


def clean_calendar(city_name: str) -> CleaningResult:
    """Clean and standardize the calendar dataset for a city.

    Calendar is the largest file (~24M rows for a city with 65K listings).
    All transformations are vectorized for performance.

    Transformations:
      1. Price columns → Float64
      2. Available → Boolean
      3. Date → native Date

    Args:
        city_name: City key from cities.yaml.

    Returns:
        CleaningResult with row counts and output paths.
    """
    config = _load_cleaning_config("calendar")
    filepath, df = _read_raw_file(city_name, "calendar")
    input_rows = df.height
    columns_cleaned: dict[str, str] = {}

    # 1. Price columns
    price_cols = config.get("price_columns", [])
    df = _clean_price_columns(df, price_cols)
    for c in price_cols:
        if c in df.columns:
            columns_cleaned[c] = "price→float64"

    # 2. Boolean columns
    bool_cols = config.get("boolean_columns", [])
    df = _cast_boolean_columns(df, bool_cols)
    for c in bool_cols:
        if c in df.columns:
            columns_cleaned[c] = "t/f→boolean"

    # 3. Date columns
    date_cols = config.get("date_columns", [])
    df = _parse_date_columns(df, date_cols)
    for c in date_cols:
        if c in df.columns:
            columns_cleaned[c] = "string→date"

    # 4. Missing value strategies
    strategies = config.get("missing_value_strategies", {})
    df, imputed_cols = _apply_missing_strategies(df, strategies)

    # 5. Validation flags & partitioning
    validation_rules = config.get("validation_rules", {})
    df = _compute_validation_flags(df, validation_rules)

    output_path, rejected_path, valid_count, rejected_count = _partition_and_save(
        df, city_name, "calendar"
    )

    logger.info(
        "Calendar cleaning complete: %d input → %d valid + %d rejected",
        input_rows,
        valid_count,
        rejected_count,
    )

    return CleaningResult(
        file_type="calendar",
        city=city_name,
        source_file=filepath.name,
        input_rows=input_rows,
        output_rows=valid_count,
        rejected_rows=rejected_count,
        output_path=str(output_path),
        rejected_path=str(rejected_path) if rejected_path else None,
        columns_cleaned=columns_cleaned,
        imputed_columns=imputed_cols,
    )


def clean_reviews(city_name: str) -> CleaningResult:
    """Clean and standardize the reviews dataset for a city.

    Transformations:
      1. Date → native Date
      2. HTML stripping from comments (if present)

    Args:
        city_name: City key from cities.yaml.

    Returns:
        CleaningResult with row counts and output paths.
    """
    config = _load_cleaning_config("reviews")
    filepath, df = _read_raw_file(city_name, "reviews")
    input_rows = df.height
    columns_cleaned: dict[str, str] = {}

    # 1. Date columns
    date_cols = config.get("date_columns", [])
    df = _parse_date_columns(df, date_cols)
    for c in date_cols:
        if c in df.columns:
            columns_cleaned[c] = "string→date"

    # 2. Special columns (HTML stripping)
    special = config.get("special_columns", {})
    df = _apply_special_columns(df, special)
    for c in special:
        if c in df.columns:
            columns_cleaned[c] = f"special:{special[c]}"

    # 3. Missing value strategies
    strategies = config.get("missing_value_strategies", {})
    df, imputed_cols = _apply_missing_strategies(df, strategies)

    # 4. Validation flags & partitioning
    validation_rules = config.get("validation_rules", {})
    df = _compute_validation_flags(df, validation_rules)

    output_path, rejected_path, valid_count, rejected_count = _partition_and_save(
        df, city_name, "reviews"
    )

    logger.info(
        "Reviews cleaning complete: %d input → %d valid + %d rejected",
        input_rows,
        valid_count,
        rejected_count,
    )

    return CleaningResult(
        file_type="reviews",
        city=city_name,
        source_file=filepath.name,
        input_rows=input_rows,
        output_rows=valid_count,
        rejected_rows=rejected_count,
        output_path=str(output_path),
        rejected_path=str(rejected_path) if rejected_path else None,
        columns_cleaned=columns_cleaned,
        imputed_columns=imputed_cols,
    )


def clean_neighbourhoods(city_name: str) -> CleaningResult:
    """Clean and standardize the neighbourhoods dataset for a city.

    Transformations:
      1. Text normalization (strip whitespace)

    Args:
        city_name: City key from cities.yaml.

    Returns:
        CleaningResult with row counts and output paths.
    """
    config = _load_cleaning_config("neighbourhoods")
    filepath, df = _read_raw_file(city_name, "neighbourhoods")
    input_rows = df.height
    columns_cleaned: dict[str, str] = {}

    # 1. Text normalization
    text_cols = config.get("text_normalize_columns", [])
    df = _normalize_text_columns(df, text_cols)
    for c in text_cols:
        if c in df.columns:
            columns_cleaned[c] = "text_normalized"

    # 2. Missing value strategies
    strategies = config.get("missing_value_strategies", {})
    df, imputed_cols = _apply_missing_strategies(df, strategies)

    # 3. Validation flags & partitioning
    validation_rules = config.get("validation_rules", {})
    df = _compute_validation_flags(df, validation_rules)

    output_path, rejected_path, valid_count, rejected_count = _partition_and_save(
        df, city_name, "neighbourhoods"
    )

    logger.info(
        "Neighbourhoods cleaning complete: %d input → %d valid + %d rejected",
        input_rows,
        valid_count,
        rejected_count,
    )

    return CleaningResult(
        file_type="neighbourhoods",
        city=city_name,
        source_file=filepath.name,
        input_rows=input_rows,
        output_rows=valid_count,
        rejected_rows=rejected_count,
        output_path=str(output_path),
        rejected_path=str(rejected_path) if rejected_path else None,
        columns_cleaned=columns_cleaned,
        imputed_columns=imputed_cols,
    )


# ===================================================================
# City-level orchestrator
# ===================================================================

# Mapping of file types to their cleaning functions
_CLEANERS: dict[str, Any] = {
    "listings": clean_listings,
    "calendar": clean_calendar,
    "reviews": clean_reviews,
    "neighbourhoods": clean_neighbourhoods,
}


def clean_city(city_name: str) -> dict[str, CleaningResult]:
    """Run the full cleaning pipeline for all files in a city.

    Cleans each file type in order: neighbourhoods → listings →
    calendar → reviews (neighbourhoods first because listings
    validation may reference neighbourhood names).

    Args:
        city_name: City key from cities.yaml.

    Returns:
        Dict mapping file_type → CleaningResult.
    """
    logger.info("=" * 60)
    logger.info("Starting cleaning pipeline for: %s", city_name)
    logger.info("=" * 60)

    # Ordered execution — neighbourhoods first, then listings, then the rest
    execution_order = ["neighbourhoods", "listings", "calendar", "reviews"]
    results: dict[str, CleaningResult] = {}

    for file_type in execution_order:
        cleaner_fn = _CLEANERS.get(file_type)
        if cleaner_fn is None:
            continue

        try:
            result = cleaner_fn(city_name)
            results[file_type] = result
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", file_type, exc)
        except Exception:
            logger.exception("Failed to clean %s for %s", file_type, city_name)

    # Save cleaning summary
    _save_cleaning_summary(city_name, results)

    total_input = sum(r.input_rows for r in results.values())
    total_output = sum(r.output_rows for r in results.values())
    total_rejected = sum(r.rejected_rows for r in results.values())

    logger.info("=" * 60)
    logger.info(
        "Cleaning complete for %s: %d → %d valid + %d rejected",
        city_name,
        total_input,
        total_output,
        total_rejected,
    )
    logger.info("=" * 60)

    return results


def _save_cleaning_summary(
    city_name: str,
    results: dict[str, CleaningResult],
) -> Path:
    """Save cleaning summary report to outputs/quality/.

    Args:
        city_name: City key.
        results: Dict of file_type → CleaningResult.

    Returns:
        Path to the saved summary JSON.
    """
    output_dir = get_output_dir("quality")
    output_path = output_dir / f"{city_name}_cleaning_summary.json"

    summary = {
        "city": city_name,
        "cleaned_at": datetime.now(UTC).isoformat(),
        "files": {file_type: asdict(result) for file_type, result in results.items()},
        "totals": {
            "input_rows": sum(r.input_rows for r in results.values()),
            "output_rows": sum(r.output_rows for r in results.values()),
            "rejected_rows": sum(r.rejected_rows for r in results.values()),
            "rejection_rate_pct": round(
                sum(r.rejected_rows for r in results.values())
                / max(sum(r.input_rows for r in results.values()), 1)
                * 100,
                2,
            ),
        },
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    logger.info("Cleaning summary saved: %s", output_path)
    return output_path
