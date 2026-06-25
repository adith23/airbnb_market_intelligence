"""Data quality validation and constraint checking for Inside Airbnb datasets.

Validates profiled data against declarative rules defined in
config/validation_rules.yaml. Produces structured quality reports
documenting constraint violations, scraping artifacts, and coverage gaps.

Outputs are written to: outputs/quality/{city}_quality_report.json

Usage:
    from src.platform.data_engineering.ingestion.validator import generate_quality_report
    report = generate_quality_report("paris")
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.platform.common.utils import (
    CONFIG_DIR,
    filter_raw_files,
    get_output_dir,
    get_raw_data_dir,
    infer_file_type,
    load_yaml_config,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Rule loading
# ===================================================================


def load_validation_rules(file_type: str | None = None) -> dict[str, Any]:
    """Load validation rules from config/validation_rules.yaml.

    Args:
        file_type: Optional file type key to return rules for a single
                   file type (e.g., 'listings'). If None, returns all.

    Returns:
        Validation rules dictionary.
    """
    rules = load_yaml_config(CONFIG_DIR / "validation_rules.yaml")

    if file_type is not None:
        return rules.get(file_type, {})

    return rules


# ===================================================================
# Column-level validation
# ===================================================================


def validate_column(
    series: pl.Series,
    col_name: str,
    rules: dict[str, Any],
) -> dict[str, Any]:
    """Validate a single column against its declared rules.

    Supports rule types:
      - not_null:  checks for null values
      - unique:    checks for duplicate values
      - positive:  checks that numeric values > 0
      - range:     checks {min, max} bounds
      - regex:     checks string pattern match
      - enum:      checks against allowed value set

    Args:
        series: Polars Series to validate.
        col_name: Column name (for reporting).
        rules: Dict of rule_type → rule_spec.

    Returns:
        Validation result dict with pass/fail per rule and violation details.
    """
    total = series.len()
    results: dict[str, Any] = {
        "column": col_name,
        "total_rows": total,
        "violations": [],
        "passed": True,
    }

    non_null = series.drop_nulls()

    # --- not_null ---
    if rules.get("not_null"):
        null_count = series.null_count()
        if null_count > 0:
            results["violations"].append(
                {
                    "rule": "not_null",
                    "violation_count": null_count,
                    "violation_pct": round(null_count / total * 100, 2),
                    "message": f"{null_count} null values found ({null_count / total * 100:.1f}%)",
                }
            )

    # --- unique ---
    if rules.get("unique"):
        dup_count = non_null.len() - non_null.n_unique()
        if dup_count > 0:
            results["violations"].append(
                {
                    "rule": "unique",
                    "violation_count": dup_count,
                    "message": f"{dup_count} duplicate values found",
                }
            )

    # --- positive ---
    if rules.get("positive") and series.dtype.is_numeric():
        non_positive = non_null.filter(non_null <= 0).len()
        if non_positive > 0:
            results["violations"].append(
                {
                    "rule": "positive",
                    "violation_count": non_positive,
                    "message": f"{non_positive} non-positive values found",
                }
            )

    # --- range ---
    range_spec = rules.get("range")
    if range_spec and series.dtype.is_numeric():
        violations = 0
        if "min" in range_spec:
            below = non_null.filter(non_null < range_spec["min"]).len()
            violations += below
        if "max" in range_spec:
            above = non_null.filter(non_null > range_spec["max"]).len()
            violations += above
        if violations > 0:
            results["violations"].append(
                {
                    "rule": "range",
                    "violation_count": violations,
                    "spec": range_spec,
                    "message": f"{violations} values outside range {range_spec}",
                }
            )

    # --- regex ---
    regex_pattern = rules.get("regex")
    if regex_pattern and (series.dtype == pl.Utf8 or series.dtype == pl.String):
        try:
            non_matching = non_null.filter(~non_null.str.contains(regex_pattern)).len()
            if non_matching > 0:
                results["violations"].append(
                    {
                        "rule": "regex",
                        "violation_count": non_matching,
                        "pattern": regex_pattern,
                        "message": f"{non_matching} values don't match pattern",
                    }
                )
        except Exception as exc:
            logger.warning("Regex validation failed for %s: %s", col_name, exc)

    # --- enum ---
    enum_values = rules.get("enum")
    if enum_values and (series.dtype == pl.Utf8 or series.dtype == pl.String):
        outside = non_null.filter(~non_null.is_in(enum_values)).len()
        if outside > 0:
            # Collect examples of unexpected values
            unexpected = non_null.filter(~non_null.is_in(enum_values)).unique().head(5).to_list()
            results["violations"].append(
                {
                    "rule": "enum",
                    "violation_count": outside,
                    "allowed_values": enum_values,
                    "unexpected_examples": unexpected,
                    "message": f"{outside} values outside allowed set",
                }
            )

    results["passed"] = len(results["violations"]) == 0
    return results


# ===================================================================
# Duplicate detection
# ===================================================================


def detect_duplicates(
    df: pl.DataFrame,
    key_columns: list[str],
) -> dict[str, Any]:
    """Detect duplicate records based on primary key column(s).

    Args:
        df: Polars DataFrame to check.
        key_columns: Column names forming the primary key.

    Returns:
        Duplicate report with counts and examples.
    """
    available_keys = [c for c in key_columns if c in df.columns]
    if not available_keys:
        return {
            "key_columns": key_columns,
            "checked": False,
            "message": "Key columns not found in DataFrame",
        }

    grouped = df.group_by(available_keys).len()
    duplicates = grouped.filter(pl.col("len") > 1)
    dup_count = duplicates.height

    result = {
        "key_columns": available_keys,
        "checked": True,
        "total_rows": df.height,
        "unique_keys": grouped.height,
        "duplicate_keys": dup_count,
        "duplicate_rows": (int(duplicates["len"].sum() - dup_count) if dup_count > 0 else 0),
    }

    if dup_count > 0:
        examples = duplicates.sort("len", descending=True).head(5)
        result["examples"] = examples.to_dicts()
        logger.warning(
            "Found %d duplicate keys on columns %s",
            dup_count,
            available_keys,
        )

    return result


# ===================================================================
# Scraping artifact detection
# ===================================================================


def detect_scraping_artifacts(df: pl.DataFrame) -> dict[str, Any]:
    """Detect common web-scraping artifacts in a DataFrame.

    Checks for:
      - HTML tags in text fields
      - Encoding artifacts (mojibake indicators)
      - Suspiciously uniform values suggesting scraper defaults

    Args:
        df: Polars DataFrame to inspect.

    Returns:
        Artifact report with counts per column.
    """
    artifacts: list[dict[str, Any]] = []

    string_cols = [
        col for col in df.columns if df[col].dtype == pl.Utf8 or df[col].dtype == pl.String
    ]

    for col_name in string_cols:
        col = df[col_name].drop_nulls()
        if col.len() == 0:
            continue

        col_artifacts: dict[str, int] = {}

        # Check for HTML tags
        html_count = col.filter(col.str.contains(r"<[a-zA-Z][^>]*>")).len()
        if html_count > 0:
            col_artifacts["html_tags"] = html_count

        # Check for common encoding artifacts
        encoding_count = col.filter(col.str.contains(r"Ã©|Ã¨|Ã |Ã§|â€™|â€œ|â€")).len()
        if encoding_count > 0:
            col_artifacts["encoding_artifacts"] = encoding_count

        if col_artifacts:
            artifacts.append(
                {
                    "column": col_name,
                    "artifacts": col_artifacts,
                }
            )

    return {
        "columns_checked": len(string_cols),
        "columns_with_artifacts": len(artifacts),
        "details": artifacts,
    }


# ===================================================================
# Coverage assessment
# ===================================================================


def assess_coverage(
    df: pl.DataFrame,
    file_type: str,
) -> dict[str, Any]:
    """Assess data coverage: completeness of key dimensions.

    For each file type, checks relevant coverage metrics:
      - listings: neighbourhood coverage, host completeness, price fill rate
      - calendar: date range span, listing coverage
      - reviews: date range span, listing coverage

    Args:
        df: Polars DataFrame.
        file_type: Logical file type ('listings', 'calendar', 'reviews').

    Returns:
        Coverage report dict.
    """
    coverage: dict[str, Any] = {"file_type": file_type}

    # Universal coverage: column-level null rates
    column_completeness = {}
    for col_name in df.columns:
        null_pct = round(df[col_name].null_count() / df.height * 100, 2)
        column_completeness[col_name] = {
            "fill_rate_pct": round(100 - null_pct, 2),
            "null_pct": null_pct,
        }
    coverage["column_completeness"] = column_completeness

    # File-type-specific checks
    if file_type == "listings":
        coverage.update(_assess_listings_coverage(df))
    elif file_type == "calendar":
        coverage.update(_assess_calendar_coverage(df))
    elif file_type == "reviews":
        coverage.update(_assess_reviews_coverage(df))

    return coverage


def _assess_listings_coverage(df: pl.DataFrame) -> dict[str, Any]:
    """Listings-specific coverage checks."""
    result: dict[str, Any] = {}

    if "neighbourhood_cleansed" in df.columns:
        result["unique_neighbourhoods"] = df["neighbourhood_cleansed"].n_unique()

    if "host_id" in df.columns:
        result["unique_hosts"] = df["host_id"].n_unique()
        result["listings_per_host_mean"] = round(df.height / df["host_id"].n_unique(), 2)

    if "price" in df.columns:
        result["price_fill_rate_pct"] = round((1 - df["price"].null_count() / df.height) * 100, 2)

    if "last_review" in df.columns:
        non_null_reviews = df["last_review"].drop_nulls()
        result["listings_with_reviews_pct"] = round(non_null_reviews.len() / df.height * 100, 2)

    return result


def _assess_calendar_coverage(df: pl.DataFrame) -> dict[str, Any]:
    """Calendar-specific coverage checks."""
    result: dict[str, Any] = {}

    if "listing_id" in df.columns:
        result["unique_listings"] = df["listing_id"].n_unique()

    if "date" in df.columns:
        dates = df["date"].drop_nulls()
        if dates.len() > 0:
            # Try to cast string dates to date type
            if dates.dtype == pl.Utf8 or dates.dtype == pl.String:
                try:
                    dates = dates.str.to_date("%Y-%m-%d")
                except Exception:
                    pass
            if dates.dtype == pl.Date:
                result["date_range"] = {
                    "min": str(dates.min()),
                    "max": str(dates.max()),
                    "span_days": (dates.max() - dates.min()).days,
                }

    if "available" in df.columns:
        avail = df["available"].drop_nulls()
        if avail.dtype == pl.Utf8 or avail.dtype == pl.String:
            available_count = avail.filter(avail == "t").len()
            unavailable_count = avail.filter(avail == "f").len()
        else:
            available_count = avail.filter(avail).len()
            unavailable_count = avail.filter(~avail).len()

        total = available_count + unavailable_count
        result["availability"] = {
            "available_pct": (round(available_count / total * 100, 2) if total > 0 else 0),
            "unavailable_pct": (round(unavailable_count / total * 100, 2) if total > 0 else 0),
        }

    return result


def _assess_reviews_coverage(df: pl.DataFrame) -> dict[str, Any]:
    """Reviews-specific coverage checks."""
    result: dict[str, Any] = {}

    if "listing_id" in df.columns:
        result["unique_listings_reviewed"] = df["listing_id"].n_unique()

    if "reviewer_id" in df.columns:
        result["unique_reviewers"] = df["reviewer_id"].n_unique()

    if "date" in df.columns:
        dates = df["date"].drop_nulls()
        if dates.len() > 0:
            if dates.dtype == pl.Utf8 or dates.dtype == pl.String:
                try:
                    dates = dates.str.to_date("%Y-%m-%d")
                except Exception:
                    pass
            if dates.dtype == pl.Date:
                result["date_range"] = {
                    "min": str(dates.min()),
                    "max": str(dates.max()),
                    "span_days": (dates.max() - dates.min()).days,
                }

    return result


# ===================================================================
# Full quality report generation
# ===================================================================


def generate_quality_report(city_name: str) -> dict[str, Any]:
    """Generate a comprehensive data quality report for a city.

    Runs all validation checks on every data file:
      1. Column-level rule validation
      2. Primary key duplicate detection
      3. Scraping artifact detection
      4. Coverage assessment

    Args:
        city_name: City key from cities.yaml.

    Returns:
        Complete quality report dict.

    Raises:
        FileNotFoundError: If raw data directory doesn't exist.
    """
    raw_dir = get_raw_data_dir(city_name)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    logger.info("Generating quality report for: %s", city_name)

    data_files = filter_raw_files(raw_dir)

    file_reports: dict[str, dict[str, Any]] = {}

    # Primary key definitions per file type
    pk_map = {
        "listings": ["id"],
        "calendar": ["listing_id", "date"],
        "reviews": ["id"],
        "neighbourhoods": ["neighbourhood"],
    }

    for filepath in data_files:
        file_type = infer_file_type(filepath)
        logger.info("Validating: %s (type: %s)", filepath.name, file_type)

        try:
            df = pl.read_csv(
                filepath,
                infer_schema_length=10_000,
                try_parse_dates=False,
                null_values=["", "N/A", "NA"],
                truncate_ragged_lines=True,
            )
        except Exception:
            logger.exception("Failed to read: %s", filepath.name)
            file_reports[filepath.name] = {"status": "READ_ERROR"}
            continue

        file_report: dict[str, Any] = {
            "file": filepath.name,
            "file_type": file_type,
            "row_count": df.height,
            "column_count": df.width,
        }

        # 1. Column-level validation
        rules = load_validation_rules(file_type)
        column_validations: dict[str, Any] = {}
        for col_name, col_rules in rules.items():
            if col_name in df.columns:
                column_validations[col_name] = validate_column(df[col_name], col_name, col_rules)
            else:
                column_validations[col_name] = {
                    "column": col_name,
                    "passed": False,
                    "violations": [{"rule": "exists", "message": "Column not found"}],
                }
        file_report["column_validations"] = column_validations

        # 2. Duplicate detection
        pk_cols = pk_map.get(file_type, [])
        file_report["duplicates"] = detect_duplicates(df, pk_cols)

        # 3. Scraping artifacts
        file_report["scraping_artifacts"] = detect_scraping_artifacts(df)

        # 4. Coverage
        file_report["coverage"] = assess_coverage(df, file_type)

        # Summary
        total_rules = len(column_validations)
        passed_rules = sum(1 for v in column_validations.values() if v.get("passed", False))
        file_report["summary"] = {
            "rules_checked": total_rules,
            "rules_passed": passed_rules,
            "rules_failed": total_rules - passed_rules,
            "has_duplicates": file_report["duplicates"].get("duplicate_keys", 0) > 0,
            "has_artifacts": file_report["scraping_artifacts"]["columns_with_artifacts"] > 0,
        }

        file_reports[filepath.name] = file_report

    # Build city-level report
    report = {
        "city": city_name,
        "generated_at": datetime.now(UTC).isoformat(),
        "files_checked": len(file_reports),
        "file_reports": file_reports,
    }

    # Save report
    output_path = _save_quality_report(city_name, report)
    logger.info("Quality report saved: %s", output_path)

    return report


def _save_quality_report(city_name: str, report: dict) -> Path:
    """Save quality report to outputs/quality/."""
    output_dir = get_output_dir("quality")
    output_path = output_dir / f"{city_name}_quality_report.json"

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    return output_path
