"""Schema discovery and statistical profiling for Inside Airbnb datasets.

This module lazily scans raw CSV/GZ files to produce two outputs:
  1. Schema JSON — column names, inferred types, nullability, sample values
  2. Profile JSON — column-level statistics (numeric, string, date)

The profiler is designed for large files (calendar.csv.gz: ~24M rows)
using Polars' lazy evaluation to minimize memory consumption.

Outputs are written to:
  - outputs/schemas/{city}_{file_type}_schema.json
  - outputs/profiles/{city}_{file_type}_profile.json

Usage:
    from src.platform.data_science.evaluation.profiler import profile_city
    profiles = profile_city("paris")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from src.platform.common.utils import (
    compute_schema_hash,
    filter_raw_files,
    get_output_dir,
    get_raw_data_dir,
    infer_file_type,
)

logger = logging.getLogger(__name__)

# Maximum sample values to include in schema output
MAX_SAMPLE_VALUES = 5
# Maximum top-N values for categorical distributions
MAX_TOP_VALUES = 10
# Row limit for initial schema inference (performance guard)
SCHEMA_INFERENCE_ROWS = 10_000


# ===================================================================
# Schema inference
# ===================================================================


def infer_schema(filepath: str | Path) -> dict[str, Any]:
    """Infer the schema of a CSV file: column names, types, nullability.

    Uses Polars to read a sample for type inference, then scans the
    full file for null counts. For gzipped files, reads the full file
    since gzip does not support random access.

    Args:
        filepath: Path to .csv or .csv.gz file.

    Returns:
        Schema dict with keys:
          file, file_type, row_count, column_count, schema_hash,
          columns: {col_name: {dtype, null_count, null_pct, sample_values}}
    """
    filepath = Path(filepath)
    file_type = infer_file_type(filepath)
    logger.info("Inferring schema for: %s (type: %s)", filepath.name, file_type)

    # Read full file — Polars handles gzip natively
    df = pl.read_csv(
        filepath,
        infer_schema_length=SCHEMA_INFERENCE_ROWS,
        try_parse_dates=False,  # Keep raw strings for profiling
        null_values=["", "N/A", "NA"],
        truncate_ragged_lines=True,
    )

    row_count = df.height
    col_count = df.width
    columns: dict[str, dict[str, Any]] = {}

    for col_name in df.columns:
        col = df[col_name]
        null_count = col.null_count()
        null_pct = round(null_count / row_count * 100, 2) if row_count > 0 else 0.0

        # Collect sample values (first N non-null)
        non_null = col.drop_nulls()
        sample_values = non_null.head(MAX_SAMPLE_VALUES).to_list() if non_null.len() > 0 else []
        # Ensure sample values are JSON-serializable
        sample_values = [_make_serializable(v) for v in sample_values]

        columns[col_name] = {
            "dtype": str(col.dtype),
            "null_count": null_count,
            "null_pct": null_pct,
            "non_null_count": row_count - null_count,
            "sample_values": sample_values,
        }

    schema = {
        "file": filepath.name,
        "file_type": file_type,
        "file_size_bytes": filepath.stat().st_size,
        "row_count": row_count,
        "column_count": col_count,
        "schema_hash": compute_schema_hash(df.columns),
        "profiled_at": datetime.now(timezone.utc).isoformat(),
        "columns": columns,
    }

    logger.info(
        "Schema inferred: %s — %d rows, %d columns",
        filepath.name,
        row_count,
        col_count,
    )
    return schema


# ===================================================================
# Column-level statistical profiling
# ===================================================================


def compute_column_stats(df: pl.DataFrame, col_name: str) -> dict[str, Any]:
    """Compute detailed statistics for a single column.

    Produces different statistics based on the column's inferred type:
      - Numeric: min, max, mean, median, std, percentiles (25th, 75th)
      - String:  min_length, max_length, avg_length, top_values
      - All:     unique_count, unique_pct

    Args:
        df: Polars DataFrame.
        col_name: Column name to profile.

    Returns:
        Statistics dictionary.
    """
    col = df[col_name]
    total = df.height
    non_null = col.drop_nulls()
    non_null_count = non_null.len()

    stats: dict[str, Any] = {
        "unique_count": non_null.n_unique() if non_null_count > 0 else 0,
        "unique_pct": (
            round(non_null.n_unique() / non_null_count * 100, 2) if non_null_count > 0 else 0.0
        ),
    }

    dtype = col.dtype

    # --- Numeric columns ---
    if dtype.is_numeric():
        stats.update(_compute_numeric_stats(non_null))

    # --- String columns ---
    elif dtype == pl.Utf8 or dtype == pl.String:
        stats.update(_compute_string_stats(non_null))

    # --- Date columns ---
    elif dtype == pl.Date or dtype == pl.Datetime:
        stats.update(_compute_date_stats(non_null))

    return stats


def _compute_numeric_stats(col: pl.Series) -> dict[str, Any]:
    """Compute numeric statistics for a non-null series.

    Args:
        col: Polars Series with numeric dtype, nulls already dropped.

    Returns:
        Dict with min, max, mean, median, std, q25, q75.
    """
    if col.len() == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "q25": None,
            "q75": None,
        }

    return {
        "min": _make_serializable(col.min()),
        "max": _make_serializable(col.max()),
        "mean": round(float(col.mean()), 4) if col.mean() is not None else None,
        "median": _make_serializable(col.median()),
        "std": round(float(col.std()), 4) if col.std() is not None else None,
        "q25": _make_serializable(col.quantile(0.25)),
        "q75": _make_serializable(col.quantile(0.75)),
    }


def _compute_string_stats(col: pl.Series) -> dict[str, Any]:
    """Compute string-specific statistics.

    Args:
        col: Polars Series with string dtype, nulls already dropped.

    Returns:
        Dict with min_length, max_length, avg_length, top_values.
    """
    if col.len() == 0:
        return {
            "min_length": None,
            "max_length": None,
            "avg_length": None,
            "top_values": {},
        }

    lengths = col.str.len_chars()

    # Top N most frequent values
    value_counts = col.value_counts(sort=True)
    top_entries = value_counts.head(MAX_TOP_VALUES)
    # value_counts returns a DataFrame with columns [column_name, "count"]
    count_col = "count" if "count" in top_entries.columns else top_entries.columns[-1]
    value_col = [c for c in top_entries.columns if c != count_col][0]
    top_values = {
        str(row[value_col]): int(row[count_col]) for row in top_entries.iter_rows(named=True)
    }

    return {
        "min_length": int(lengths.min()) if lengths.min() is not None else None,
        "max_length": int(lengths.max()) if lengths.max() is not None else None,
        "avg_length": (round(float(lengths.mean()), 1) if lengths.mean() is not None else None),
        "top_values": top_values,
    }


def _compute_date_stats(col: pl.Series) -> dict[str, Any]:
    """Compute date range statistics.

    Args:
        col: Polars Series with date/datetime dtype, nulls dropped.

    Returns:
        Dict with min_date, max_date, date_range_days.
    """
    if col.len() == 0:
        return {"min_date": None, "max_date": None, "date_range_days": None}

    min_date = col.min()
    max_date = col.max()
    range_days = (max_date - min_date).days if min_date and max_date else None

    return {
        "min_date": str(min_date) if min_date else None,
        "max_date": str(max_date) if max_date else None,
        "date_range_days": range_days,
    }


# ===================================================================
# File-level profiling
# ===================================================================


def profile_file(filepath: str | Path) -> dict[str, Any]:
    """Generate a full profile for a single data file.

    Combines schema inference with column-level statistics into
    a single comprehensive profile document.

    Args:
        filepath: Path to .csv or .csv.gz file.

    Returns:
        Complete profile dict with schema and per-column statistics.
    """
    filepath = Path(filepath)
    logger.info("Profiling file: %s", filepath.name)

    # Step 1: Infer schema
    schema = infer_schema(filepath)

    # Step 2: Read data for statistics
    df = pl.read_csv(
        filepath,
        infer_schema_length=SCHEMA_INFERENCE_ROWS,
        try_parse_dates=True,
        null_values=["", "N/A", "NA"],
        truncate_ragged_lines=True,
    )

    # Step 3: Compute per-column statistics
    column_profiles: dict[str, dict[str, Any]] = {}
    for col_name in df.columns:
        col_stats = compute_column_stats(df, col_name)

        # Merge schema info with stats
        col_schema = schema["columns"].get(col_name, {})
        column_profiles[col_name] = {**col_schema, **col_stats}

    profile = {
        "file": filepath.name,
        "file_type": schema["file_type"],
        "file_size_bytes": schema["file_size_bytes"],
        "row_count": schema["row_count"],
        "column_count": schema["column_count"],
        "schema_hash": schema["schema_hash"],
        "profiled_at": schema["profiled_at"],
        "columns": column_profiles,
    }

    logger.info(
        "Profile complete: %s — %d rows × %d columns",
        filepath.name,
        profile["row_count"],
        profile["column_count"],
    )
    return profile


# ===================================================================
# City-level profiling
# ===================================================================


def profile_city(city_name: str) -> dict[str, dict[str, Any]]:
    """Profile all data files for a city.

    Discovers all CSV/GZ files in data/raw/{city_name}/, profiles each,
    and saves individual schema and profile JSONs to the outputs directory.

    Args:
        city_name: City key from cities.yaml.

    Returns:
        Dict mapping file_type → profile dict.

    Raises:
        FileNotFoundError: If the city's raw data directory does not exist.
    """
    raw_dir = get_raw_data_dir(city_name)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}. "
            f"Run the downloader first: python main.py download --city {city_name}"
        )

    # Discover data files
    data_files = filter_raw_files(raw_dir)

    if not data_files:
        logger.warning("No CSV files found in %s", raw_dir)
        return {}

    logger.info("Profiling %d files for city: %s", len(data_files), city_name)

    profiles: dict[str, dict[str, Any]] = {}

    for filepath in data_files:
        file_type = infer_file_type(filepath)

        try:
            profile = profile_file(filepath)
            profiles[filepath.name] = profile

            # Save schema
            save_schema(
                profile,
                city_name=city_name,
                file_type=file_type,
                filename=filepath.name,
            )

            # Save profile
            save_profile(
                profile,
                city_name=city_name,
                file_type=file_type,
                filename=filepath.name,
            )

        except Exception:
            logger.exception("Failed to profile: %s", filepath.name)

    logger.info("City profiling complete: %s (%d files)", city_name, len(profiles))
    return profiles


# ===================================================================
# Output persistence
# ===================================================================


def save_schema(
    profile: dict[str, Any],
    city_name: str,
    file_type: str,
    filename: str,
) -> Path:
    """Save schema information to outputs/schemas/.

    Extracts only schema-relevant fields (no statistics) for a
    clean schema document.

    Args:
        profile: Full profile dict from profile_file().
        city_name: City key.
        file_type: Logical file type.
        filename: Original filename.

    Returns:
        Path to the saved schema JSON.
    """
    schema_dir = get_output_dir("schemas")

    # Extract schema-only fields (exclude stats)
    schema_doc = {
        "file": filename,
        "file_type": file_type,
        "city": city_name,
        "row_count": profile["row_count"],
        "column_count": profile["column_count"],
        "schema_hash": profile["schema_hash"],
        "profiled_at": profile["profiled_at"],
        "columns": {
            col_name: {
                "dtype": col_info.get("dtype"),
                "null_count": col_info.get("null_count"),
                "null_pct": col_info.get("null_pct"),
                "sample_values": col_info.get("sample_values", []),
            }
            for col_name, col_info in profile["columns"].items()
        },
    }

    # Use descriptive filename to distinguish summary vs detailed
    safe_name = filename.replace(".csv.gz", "_detailed").replace(".csv", "_summary")
    output_path = schema_dir / f"{city_name}_{safe_name}_schema.json"

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(schema_doc, fh, indent=2, default=str)

    logger.info("Schema saved: %s", output_path.name)
    return output_path


def save_profile(
    profile: dict[str, Any],
    city_name: str,
    file_type: str,
    filename: str,
) -> Path:
    """Save full statistical profile to outputs/profiles/.

    Args:
        profile: Full profile dict from profile_file().
        city_name: City key.
        file_type: Logical file type.
        filename: Original filename.

    Returns:
        Path to the saved profile JSON.
    """
    profile_dir = get_output_dir("profiles")

    safe_name = filename.replace(".csv.gz", "_detailed").replace(".csv", "_summary")
    output_path = profile_dir / f"{city_name}_{safe_name}_profile.json"

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2, default=str)

    logger.info("Profile saved: %s", output_path.name)
    return output_path


# ===================================================================
# Helpers
# ===================================================================


def _make_serializable(value: Any) -> Any:
    """Convert a value to a JSON-serializable type.

    Handles Polars-specific types, numpy types, dates, and other
    non-standard types that json.dumps cannot serialize.

    Args:
        value: Any value.

    Returns:
        JSON-serializable equivalent.
    """
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_make_serializable(v) for v in value]

    # Attempt numeric conversion for numpy/polars numeric types
    try:
        return float(value)
    except (TypeError, ValueError):
        pass

    return str(value)


# ===================================================================
# Outlier detection (IQR method)
# ===================================================================

# Default columns to check for outliers per file type
_OUTLIER_TARGETS: dict[str, list[str]] = {
    "listings": [
        "price",
        "minimum_nights",
        "maximum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "availability_365",
        "accommodates",
        "calculated_host_listings_count",
    ],
    "calendar": ["price", "minimum_nights", "maximum_nights"],
    "reviews": [],
    "neighbourhoods": [],
}


def detect_outliers_iqr(
    df: pl.DataFrame,
    col_name: str,
    multiplier: float = 1.5,
) -> dict[str, Any]:
    """Detect outliers in a numeric column using the IQR method.

    IQR (Interquartile Range) is robust to skewed distributions,
    which is characteristic of Airbnb pricing data.

    Bounds:
        lower = Q1 - multiplier × IQR
        upper = Q3 + multiplier × IQR

    Args:
        df: Polars DataFrame.
        col_name: Numeric column to check.
        multiplier: IQR multiplier (default 1.5; use 3.0 for extreme only).

    Returns:
        Outlier report dict with bounds, counts, and examples.
    """
    if col_name not in df.columns:
        return {"column": col_name, "error": "Column not found"}

    col = df[col_name].drop_nulls()

    if not col.dtype.is_numeric() or col.len() == 0:
        return {"column": col_name, "error": "Not numeric or empty"}

    q1 = float(col.quantile(0.25))
    q3 = float(col.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_mask = (col < lower_bound) | (col > upper_bound)
    outlier_count = outlier_mask.sum()
    total = col.len()

    # Collect example outlier values (up to 10)
    outlier_values = col.filter(outlier_mask)
    examples = sorted([_make_serializable(v) for v in outlier_values.head(10).to_list()])

    return {
        "column": col_name,
        "total_non_null": total,
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "multiplier": multiplier,
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "outlier_count": int(outlier_count),
        "outlier_pct": (round(float(outlier_count) / total * 100, 2) if total > 0 else 0.0),
        "outlier_examples": examples,
    }


def detect_outliers_for_file(
    df: pl.DataFrame,
    file_type: str,
) -> list[dict[str, Any]]:
    """Run IQR outlier detection on all target columns for a file type.

    Args:
        df: Polars DataFrame.
        file_type: Logical file type ('listings', 'calendar', etc.).

    Returns:
        List of per-column outlier report dicts.
    """
    target_cols = _OUTLIER_TARGETS.get(file_type, [])
    results = []

    for col_name in target_cols:
        if col_name in df.columns and df[col_name].dtype.is_numeric():
            result = detect_outliers_iqr(df, col_name)
            results.append(result)
            if result.get("outlier_count", 0) > 0:
                logger.info(
                    "Outliers in %s.%s: %d (%.1f%%)",
                    file_type,
                    col_name,
                    result["outlier_count"],
                    result["outlier_pct"],
                )

    return results


# ===================================================================
# Consolidated data quality report
# ===================================================================


def generate_data_quality_report(city_name: str) -> dict[str, Any]:
    """Generate a consolidated data quality report for a city.

    Combines profiling, outlier detection, and completeness analysis
    into a single executive report. This is the primary deliverable
    of Section 3.1.

    Args:
        city_name: City key from cities.yaml.

    Returns:
        Consolidated quality report dict with executive summary.

    Raises:
        FileNotFoundError: If raw data directory doesn't exist.
    """
    raw_dir = get_raw_data_dir(city_name)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    logger.info("Generating consolidated quality report for: %s", city_name)

    data_files = filter_raw_files(raw_dir)

    file_reports: dict[str, dict[str, Any]] = {}
    total_rows = 0
    total_columns = 0
    critical_issues: list[str] = []

    for filepath in data_files:
        file_type = infer_file_type(filepath)
        logger.info("Quality analysis: %s (type: %s)", filepath.name, file_type)

        try:
            # Profile the file
            profile = profile_file(filepath)
            total_rows += profile["row_count"]
            total_columns += profile["column_count"]

            # Read data for additional checks
            df = pl.read_csv(
                filepath,
                infer_schema_length=SCHEMA_INFERENCE_ROWS,
                try_parse_dates=False,
                null_values=["", "N/A", "NA"],
                truncate_ragged_lines=True,
            )

            # Outlier detection
            outliers = detect_outliers_for_file(df, file_type)

            # Completeness ranking (sorted by null rate descending)
            completeness = []
            for col_name in df.columns:
                null_pct = round(df[col_name].null_count() / df.height * 100, 2)
                completeness.append(
                    {
                        "column": col_name,
                        "fill_rate_pct": round(100 - null_pct, 2),
                        "null_pct": null_pct,
                        "null_count": df[col_name].null_count(),
                    }
                )
            completeness.sort(key=lambda x: x["null_pct"], reverse=True)

            # Flag critical issues
            high_null_cols = [
                c
                for c in completeness
                if c["null_pct"] > 50 and c["column"] in ("price", "latitude", "longitude", "id")
            ]
            if high_null_cols:
                for c in high_null_cols:
                    critical_issues.append(
                        f"{filepath.name}: {c['column']} is {c['null_pct']}% null"
                    )

            file_reports[filepath.name] = {
                "file_type": file_type,
                "row_count": profile["row_count"],
                "column_count": profile["column_count"],
                "profile": profile,
                "outliers": outliers,
                "completeness": completeness,
            }

        except Exception:
            logger.exception("Failed quality analysis: %s", filepath.name)
            file_reports[filepath.name] = {"status": "ERROR"}

    # Compute overall quality score (simple heuristic)
    quality_scores = []
    for _name, fr in file_reports.items():
        if fr.get("status") == "ERROR":
            continue
        completeness_list = fr.get("completeness", [])
        if completeness_list:
            avg_fill = sum(c["fill_rate_pct"] for c in completeness_list) / len(completeness_list)
            quality_scores.append(avg_fill)

    overall_score = round(sum(quality_scores) / len(quality_scores), 1) if quality_scores else 0.0

    report = {
        "city": city_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "executive_summary": {
            "total_files": len(file_reports),
            "total_rows": total_rows,
            "total_columns": total_columns,
            "overall_quality_score": overall_score,
            "critical_issues": critical_issues,
        },
        "file_reports": file_reports,
    }

    # Save report
    output_dir = get_output_dir("quality")
    output_path = output_dir / f"{city_name}_data_quality_report.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    logger.info("Quality report saved: %s (score: %.1f)", output_path.name, overall_score)
    return report
