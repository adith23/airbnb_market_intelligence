"""Cross-city schema comparison and harmonization analysis.

Compares profiled schemas across multiple cities to identify structural
differences (column presence, naming, types, coverage) and produces
a harmonization report documenting the strategy for unification.

This module is documentation-focused for Section 2.3. The actual
data transformation engine is deferred to Section 2.4.

Outputs are written to: outputs/harmonization/

Usage:
    from src.platform.data_engineering.modeling.harmonizer import generate_harmonization_report
    report = generate_harmonization_report(["paris", "new_york_city"])
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.platform.common.utils import (
    get_output_dir,
    get_raw_data_dir,
    load_city_config,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Schema loading from profiler outputs
# ===================================================================


def _load_city_schemas(city_name: str) -> dict[str, dict[str, Any]]:
    """Load profiled schemas for a city from outputs/schemas/.

    Args:
        city_name: City key from cities.yaml.

    Returns:
        Dict mapping filename → schema dict.

    Raises:
        FileNotFoundError: If no schemas found for the city.
    """
    schema_dir = get_output_dir("schemas")
    schema_files = list(schema_dir.glob(f"{city_name}_*_schema.json"))

    if not schema_files:
        raise FileNotFoundError(
            f"No profiled schemas found for '{city_name}' in {schema_dir}. "
            f"Run the profiler first: python main.py profile --city {city_name}"
        )

    schemas: dict[str, dict[str, Any]] = {}
    for filepath in schema_files:
        with open(filepath, encoding="utf-8") as fh:
            schema = json.load(fh)
            schemas[schema.get("file", filepath.stem)] = schema

    logger.info("Loaded %d schemas for city: %s", len(schemas), city_name)
    return schemas


# ===================================================================
# Schema comparison
# ===================================================================


def compare_schemas(
    city_schemas: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Compare column schemas across multiple cities.

    For each logical file type (listings, calendar, reviews, neighbourhoods),
    analyzes:
      - Column presence across cities (union, intersection, city-specific)
      - Data type differences for shared columns
      - Null rate differences
      - Row count and file size differences

    Args:
        city_schemas: Dict mapping city_name → {filename → schema_dict}.

    Returns:
        Comparison report dict.
    """
    cities = list(city_schemas.keys())
    logger.info("Comparing schemas across %d cities: %s", len(cities), cities)

    # Group schemas by file_type across cities
    by_file_type: dict[str, dict[str, dict]] = {}
    for city, schemas in city_schemas.items():
        for filename, schema in schemas.items():
            file_type = schema.get("file_type", "unknown")
            if file_type not in by_file_type:
                by_file_type[file_type] = {}
            by_file_type[file_type][city] = schema

    comparisons: dict[str, Any] = {}

    for file_type, city_file_schemas in by_file_type.items():
        comparison = _compare_file_type_schemas(file_type, city_file_schemas)
        comparisons[file_type] = comparison

    return {
        "cities_compared": cities,
        "file_types_compared": list(comparisons.keys()),
        "comparisons": comparisons,
    }


def _compare_file_type_schemas(
    file_type: str,
    city_schemas: dict[str, dict],
) -> dict[str, Any]:
    """Compare schemas for a single file type across cities.

    Args:
        file_type: Logical file type (e.g., 'listings').
        city_schemas: Dict mapping city_name → schema dict for this file type.

    Returns:
        Comparison result for this file type.
    """
    cities = list(city_schemas.keys())

    # Collect column sets per city
    city_columns: dict[str, set[str]] = {}
    for city, schema in city_schemas.items():
        columns = schema.get("columns", {})
        city_columns[city] = set(columns.keys())

    # Set analysis
    all_columns = set()
    for cols in city_columns.values():
        all_columns |= cols

    common_columns = set.intersection(*city_columns.values()) if city_columns else set()

    city_specific: dict[str, list[str]] = {}
    for city, cols in city_columns.items():
        specific = cols - common_columns
        if specific:
            city_specific[city] = sorted(specific)

    # Column-level comparison for common columns
    column_diffs: list[dict[str, Any]] = []
    for col_name in sorted(common_columns):
        col_diff = {"column": col_name, "consistent": True, "details": {}}

        dtypes_seen: dict[str, str] = {}
        null_pcts: dict[str, float] = {}

        for city, schema in city_schemas.items():
            col_info = schema.get("columns", {}).get(col_name, {})
            dtype = col_info.get("dtype", "unknown")
            null_pct = col_info.get("null_pct", 0.0)

            dtypes_seen[city] = dtype
            null_pcts[city] = null_pct

        # Check type consistency
        unique_dtypes = set(dtypes_seen.values())
        if len(unique_dtypes) > 1:
            col_diff["consistent"] = False
            col_diff["type_mismatch"] = dtypes_seen

        col_diff["details"] = {
            "dtypes": dtypes_seen,
            "null_pcts": null_pcts,
        }

        if not col_diff["consistent"]:
            column_diffs.append(col_diff)

    # Size comparison
    size_comparison: dict[str, dict[str, Any]] = {}
    for city, schema in city_schemas.items():
        size_comparison[city] = {
            "row_count": schema.get("row_count", 0),
            "column_count": schema.get("column_count", 0),
            "file_size_bytes": schema.get("file_size_bytes", 0),
        }

    return {
        "file_type": file_type,
        "cities": cities,
        "total_unique_columns": len(all_columns),
        "common_columns": len(common_columns),
        "common_column_names": sorted(common_columns),
        "city_specific_columns": city_specific,
        "type_inconsistencies": column_diffs,
        "size_comparison": size_comparison,
    }


# ===================================================================
# Dataset metadata comparison
# ===================================================================


def compare_dataset_metadata(city_names: list[str]) -> dict[str, Any]:
    """Compare high-level dataset metadata across cities.

    Compares: currency, timezone, scrape date, admin unit structure,
    file sizes, and row counts from the raw data directory.

    Args:
        city_names: List of city keys from cities.yaml.

    Returns:
        Metadata comparison dict.
    """
    comparisons: dict[str, Any] = {}

    for city_name in city_names:
        try:
            config = load_city_config(city_name)
        except KeyError:
            logger.warning("City config not found: %s", city_name)
            continue

        raw_dir = get_raw_data_dir(city_name)
        file_sizes: dict[str, int] = {}
        if raw_dir.exists():
            for f in raw_dir.iterdir():
                if f.is_file():
                    file_sizes[f.name] = f.stat().st_size

        comparisons[city_name] = {
            "display_name": config.get("display_name", city_name),
            "currency_code": config.get("currency_code"),
            "currency_symbol": config.get("currency_symbol"),
            "timezone": config.get("timezone"),
            "scrape_date": config.get("scrape_date"),
            "admin_unit_name": config.get("admin_unit_name"),
            "data_available": raw_dir.exists(),
            "file_sizes": file_sizes,
            "total_size_bytes": sum(file_sizes.values()),
        }

    return {
        "cities": city_names,
        "metadata": comparisons,
    }


# ===================================================================
# Harmonization strategy documentation
# ===================================================================


def document_harmonization_strategy(
    schema_comparison: dict[str, Any],
    metadata_comparison: dict[str, Any],
) -> dict[str, Any]:
    """Generate a structured harmonization strategy document.

    Based on the schema and metadata comparisons, produces a strategy
    document covering:
      1. Column renaming/mapping decisions
      2. Type coercion plan
      3. Currency handling strategy
      4. Missing column handling
      5. Text encoding normalization

    This is a documentation artifact — not executable transformation code.
    The actual transformations are implemented in Section 2.4.

    Args:
        schema_comparison: Output of compare_schemas().
        metadata_comparison: Output of compare_dataset_metadata().

    Returns:
        Harmonization strategy dict.
    """
    strategies: dict[str, Any] = {}

    # 1. Column mapping strategy
    strategies["column_mapping"] = {
        "approach": "Canonical schema defined in config/schema_map.yaml",
        "rules": [
            "All cities map to the same canonical column names",
            "City-specific columns are preserved with null fill for other cities",
            "US/UK spelling variants normalized (neighborhood → neighbourhood)",
        ],
    }

    # 2. Type coercion strategy
    comparisons = schema_comparison.get("comparisons", {})
    type_issues = []
    for file_type, comp in comparisons.items():
        for diff in comp.get("type_inconsistencies", []):
            type_issues.append(
                {
                    "file_type": file_type,
                    "column": diff["column"],
                    "types_found": diff.get("type_mismatch", {}),
                }
            )

    strategies["type_coercion"] = {
        "approach": "Cast to canonical types defined in schema_map.yaml",
        "issues_found": type_issues,
        "rules": [
            "String price fields → Float64 after currency symbol removal",
            "Boolean t/f fields → native Boolean",
            "Date strings → Date type with strict YYYY-MM-DD parsing",
            "Percentage strings → Float64 proportion (0.0-1.0)",
        ],
    }

    # 3. Currency strategy
    metadata = metadata_comparison.get("metadata", {})
    currencies = {city: info.get("currency_code") for city, info in metadata.items()}

    strategies["currency"] = {
        "approach": "Dual-column: price_local + price_usd",
        "currencies_found": currencies,
        "rules": [
            "Preserve original price in price_local column",
            "Add price_usd using exchange rate at scrape_date",
            "Exchange rates sourced from config/exchange_rates.csv",
            "All cross-city price comparisons use price_usd",
        ],
    }

    # 4. Missing column strategy
    missing_cols: dict[str, Any] = {}
    for file_type, comp in comparisons.items():
        specific = comp.get("city_specific_columns", {})
        if specific:
            missing_cols[file_type] = specific

    strategies["missing_columns"] = {
        "approach": "NULL fill for absent columns",
        "city_specific_columns": missing_cols,
        "rules": [
            "Canonical schema is the UNION of all city schemas",
            "Columns absent in a city's data are filled with NULL",
            "No data fabrication or cross-city imputation",
        ],
    }

    # 5. Text encoding strategy
    strategies["text_encoding"] = {
        "approach": "Normalize all text to UTF-8 NFC",
        "rules": [
            "Read with UTF-8 encoding, fallback to Latin-1 for older scrapes",
            "Normalize Unicode to NFC form",
            "Strip HTML tags from description and comment fields",
            "Preserve original language (no translation)",
        ],
    }

    # 6. Metadata enrichment
    strategies["metadata_enrichment"] = {
        "approach": "Add city and scrape context columns",
        "columns_added": [
            "city (VARCHAR) — city key from config",
            "country (VARCHAR) — country from config",
            "currency_code (VARCHAR) — original currency",
            "scrape_date (DATE) — dataset snapshot date",
        ],
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cities": schema_comparison.get("cities_compared", []),
        "strategies": strategies,
    }


# ===================================================================
# Full harmonization report
# ===================================================================


def generate_harmonization_report(city_names: list[str]) -> dict[str, Any]:
    """Generate a complete cross-city harmonization report.

    Orchestrates:
      1. Load profiled schemas for each city
      2. Compare schemas across cities
      3. Compare dataset metadata
      4. Document harmonization strategy

    Args:
        city_names: List of city keys to compare.

    Returns:
        Complete harmonization report.

    Raises:
        ValueError: If fewer than 2 cities provided.
    """
    if len(city_names) < 2:
        raise ValueError(f"Harmonization comparison requires at least 2 cities. Got: {city_names}")

    logger.info(
        "Generating harmonization report for %d cities: %s",
        len(city_names),
        city_names,
    )

    # 1. Load schemas
    all_schemas: dict[str, dict[str, dict[str, Any]]] = {}
    for city in city_names:
        try:
            all_schemas[city] = _load_city_schemas(city)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", city, exc)

    if len(all_schemas) < 2:
        raise ValueError(
            f"Need schemas for at least 2 cities. Only found: {list(all_schemas.keys())}. "
            "Run the profiler for each city first."
        )

    # 2. Compare schemas
    schema_comparison = compare_schemas(all_schemas)

    # 3. Compare metadata
    metadata_comparison = compare_dataset_metadata(city_names)

    # 4. Document strategy
    strategy = document_harmonization_strategy(schema_comparison, metadata_comparison)

    # Build complete report
    report = {
        "type": "harmonization_report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cities": city_names,
        "schema_comparison": schema_comparison,
        "metadata_comparison": metadata_comparison,
        "harmonization_strategy": strategy,
    }

    # Save outputs
    _save_harmonization_report(report)
    _save_harmonization_summary_md(report)

    logger.info("Harmonization report complete for: %s", city_names)
    return report


# ===================================================================
# Output persistence
# ===================================================================


def _save_harmonization_report(report: dict) -> Path:
    """Save harmonization report as JSON."""
    output_dir = get_output_dir("harmonization")
    output_path = output_dir / "harmonization_report.json"

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    logger.info("Harmonization report saved: %s", output_path)
    return output_path


def _save_harmonization_summary_md(report: dict) -> Path:
    """Save a human-readable harmonization summary as Markdown."""
    output_dir = get_output_dir("harmonization")
    output_path = output_dir / "harmonization_summary.md"

    cities = report.get("cities", [])
    strategy = report.get("harmonization_strategy", {}).get("strategies", {})
    comparisons = report.get("schema_comparison", {}).get("comparisons", {})
    metadata = report.get("metadata_comparison", {}).get("metadata", {})

    lines = [
        "# Cross-City Harmonization Summary",
        "",
        f"**Cities compared:** {', '.join(cities)}",
        f"**Generated:** {report.get('generated_at', 'N/A')}",
        "",
        "---",
        "",
        "## Dataset Size Comparison",
        "",
        "| City | Scrape Date | Currency | Total Data Size |",
        "|:-----|:------------|:---------|:----------------|",
    ]

    for city, meta in metadata.items():
        size_mb = round(meta.get("total_size_bytes", 0) / 1024 / 1024, 1)
        lines.append(
            f"| {meta.get('display_name', city)} "
            f"| {meta.get('scrape_date', 'N/A')} "
            f"| {meta.get('currency_code', 'N/A')} ({meta.get('currency_symbol', '')}) "
            f"| {size_mb} MB |"
        )

    lines.extend(["", "## Schema Comparison by File Type", ""])

    for file_type, comp in comparisons.items():
        lines.append(f"### {file_type.title()}")
        lines.append("")
        lines.append(f"- **Common columns:** {comp.get('common_columns', 0)}")
        lines.append(f"- **Total unique columns:** {comp.get('total_unique_columns', 0)}")

        specific = comp.get("city_specific_columns", {})
        if specific:
            lines.append("- **City-specific columns:**")
            for city, cols in specific.items():
                lines.append(f"  - {city}: `{'`, `'.join(cols)}`")

        type_issues = comp.get("type_inconsistencies", [])
        if type_issues:
            lines.append(f"- **Type inconsistencies:** {len(type_issues)} columns")
            for issue in type_issues[:5]:
                lines.append(f"  - `{issue['column']}`: {issue.get('type_mismatch', {})}")

        # Size comparison
        sizes = comp.get("size_comparison", {})
        if sizes:
            lines.append("- **Row counts:**")
            for city, info in sizes.items():
                lines.append(f"  - {city}: {info.get('row_count', 'N/A'):,} rows")

        lines.append("")

    # Strategy summary
    lines.extend(
        [
            "## Harmonization Strategy",
            "",
        ]
    )

    for strategy_name, strategy_info in strategy.items():
        title = strategy_name.replace("_", " ").title()
        approach = strategy_info.get("approach", "N/A")
        lines.append(f"### {title}")
        lines.append(f"**Approach:** {approach}")
        lines.append("")

        rules = strategy_info.get("rules", [])
        if rules:
            for rule in rules:
                lines.append(f"- {rule}")
            lines.append("")

    content = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    logger.info("Harmonization summary saved: %s", output_path)
    return output_path
