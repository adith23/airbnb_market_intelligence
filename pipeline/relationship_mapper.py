"""Primary/foreign key relationship mapping and referential integrity validation.

Identifies PK/FK relationships between Inside Airbnb dataset files,
validates referential integrity (orphan detection, join coverage),
and generates ERD documentation in Mermaid format.

Outputs are written to: outputs/relationships/

Usage:
    from pipeline.relationship_mapper import generate_relationship_report
    report = generate_relationship_report("paris")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pipeline.utils import (
    get_output_dir,
    get_raw_data_dir,
    infer_file_type,
    filter_raw_files,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Relationship definitions
# ===================================================================

# Known relationships between Inside Airbnb files.
# Each relationship defines a parent (PK side) and child (FK side).
KNOWN_RELATIONSHIPS: list[dict[str, Any]] = [
    {
        "name": "listing_to_calendar",
        "description": "Each listing has ~365 calendar entries",
        "parent": {"file_type": "listings", "key": "id"},
        "child": {"file_type": "calendar", "key": "listing_id"},
        "cardinality": "1:N",
    },
    {
        "name": "listing_to_reviews",
        "description": "Each listing receives zero or more reviews",
        "parent": {"file_type": "listings", "key": "id"},
        "child": {"file_type": "reviews", "key": "listing_id"},
        "cardinality": "1:N",
    },
    {
        "name": "neighbourhood_to_listing",
        "description": "Each neighbourhood contains zero or more listings",
        "parent": {"file_type": "neighbourhoods", "key": "neighbourhood"},
        "child": {"file_type": "listings", "key": "neighbourhood_cleansed"},
        "cardinality": "1:N",
    },
]

# Primary key definitions per file type
PRIMARY_KEYS: dict[str, list[str]] = {
    "listings": ["id"],
    "calendar": ["listing_id", "date"],
    "reviews": ["id"],
    "neighbourhoods": ["neighbourhood"],
}


# ===================================================================
# Primary key validation
# ===================================================================

def validate_primary_keys(
    df: pl.DataFrame,
    key_columns: list[str],
    file_type: str,
) -> dict[str, Any]:
    """Validate that primary key columns are unique and non-null.

    Args:
        df: Polars DataFrame.
        key_columns: Columns forming the primary key.
        file_type: Logical file type for reporting.

    Returns:
        PK validation result dict.
    """
    available_keys = [c for c in key_columns if c in df.columns]
    if not available_keys:
        return {
            "file_type": file_type,
            "key_columns": key_columns,
            "valid": False,
            "error": "Key columns not found in DataFrame",
        }

    total = df.height

    # Check for nulls in key columns
    null_counts = {
        col: df[col].null_count() for col in available_keys
    }
    has_nulls = any(count > 0 for count in null_counts.values())

    # Check uniqueness
    unique_count = df.select(available_keys).unique().height
    is_unique = unique_count == total
    duplicate_count = total - unique_count

    result = {
        "file_type": file_type,
        "key_columns": available_keys,
        "total_rows": total,
        "unique_keys": unique_count,
        "duplicate_keys": duplicate_count,
        "null_counts": null_counts,
        "has_nulls": has_nulls,
        "is_unique": is_unique,
        "valid": is_unique and not has_nulls,
    }

    if not result["valid"]:
        issues = []
        if has_nulls:
            issues.append(f"Null values in key columns: {null_counts}")
        if not is_unique:
            issues.append(f"{duplicate_count} duplicate key combinations")
        result["issues"] = issues
        logger.warning("PK validation failed for %s: %s", file_type, issues)

    return result


# ===================================================================
# Foreign key / referential integrity validation
# ===================================================================

def validate_referential_integrity(
    parent_df: pl.DataFrame,
    child_df: pl.DataFrame,
    parent_key: str,
    child_key: str,
    relationship_name: str,
) -> dict[str, Any]:
    """Check that FK values in child exist in parent's PK.

    Computes:
      - Join coverage (% of child FK values found in parent PK)
      - Orphan count (child FK values NOT in parent PK)
      - Examples of orphan values

    Args:
        parent_df: DataFrame containing the primary key.
        child_df: DataFrame containing the foreign key.
        parent_key: Column name of the PK in parent.
        child_key: Column name of the FK in child.
        relationship_name: Human-readable name for reporting.

    Returns:
        Referential integrity report dict.
    """
    if parent_key not in parent_df.columns:
        return {"relationship": relationship_name, "error": f"Parent key '{parent_key}' not found"}
    if child_key not in child_df.columns:
        return {"relationship": relationship_name, "error": f"Child key '{child_key}' not found"}

    parent_values = parent_df[parent_key].drop_nulls().unique()
    child_values = child_df[child_key].drop_nulls().unique()

    parent_set = set(parent_values.to_list())
    child_set = set(child_values.to_list())

    matched = child_set & parent_set
    orphans = child_set - parent_set

    child_unique_count = len(child_set)
    coverage_pct = (
        round(len(matched) / child_unique_count * 100, 2)
        if child_unique_count > 0
        else 100.0
    )

    # Count total orphan rows (not just unique orphan keys)
    if orphans:
        orphan_row_count = child_df.filter(
            pl.col(child_key).is_in(list(orphans))
        ).height
    else:
        orphan_row_count = 0

    result = {
        "relationship": relationship_name,
        "parent_key": parent_key,
        "child_key": child_key,
        "parent_unique_values": len(parent_set),
        "child_unique_values": child_unique_count,
        "matched_values": len(matched),
        "orphan_values": len(orphans),
        "orphan_rows": orphan_row_count,
        "coverage_pct": coverage_pct,
        "is_valid": len(orphans) == 0,
    }

    if orphans:
        # Sample orphan values for investigation
        result["orphan_examples"] = sorted(list(orphans))[:10]
        logger.warning(
            "Referential integrity issue in '%s': %d orphan FK values (%.1f%% coverage)",
            relationship_name, len(orphans), coverage_pct,
        )
    else:
        logger.info(
            "Referential integrity OK for '%s': 100%% coverage",
            relationship_name,
        )

    return result


# ===================================================================
# City-level relationship analysis
# ===================================================================

def generate_relationship_report(city_name: str) -> dict[str, Any]:
    """Generate a comprehensive relationship and integrity report for a city.

    Validates:
      1. Primary key constraints on each file
      2. Foreign key referential integrity across files
      3. Join coverage statistics

    Also generates a Mermaid ERD string for documentation.

    Args:
        city_name: City key from cities.yaml.

    Returns:
        Relationship report dict.

    Raises:
        FileNotFoundError: If raw data directory doesn't exist.
    """
    raw_dir = get_raw_data_dir(city_name)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    logger.info("Generating relationship report for: %s", city_name)

    # Load all data files into a dict keyed by file_type
    dataframes: dict[str, pl.DataFrame] = {}
    data_files = filter_raw_files(raw_dir)

    for filepath in data_files:
        file_type = infer_file_type(filepath)
        if file_type == "unknown":
            continue

        # For relationship mapping, we only need key columns — not full data
        # But for accuracy, load full file (Polars handles it efficiently)
        try:
            df = pl.read_csv(
                filepath,
                infer_schema_length=10_000,
                try_parse_dates=False,
                null_values=["", "N/A", "NA"],
                truncate_ragged_lines=True,
            )

            # If multiple files of same type (summary vs detailed),
            # prefer the larger one (detailed)
            existing = dataframes.get(file_type)
            if existing is None or df.width > existing.width:
                dataframes[file_type] = df
                logger.info(
                    "Loaded %s: %s (%d rows × %d cols)",
                    file_type, filepath.name, df.height, df.width,
                )
        except Exception:
            logger.exception("Failed to load: %s", filepath.name)

    # 1. Validate primary keys
    pk_results: dict[str, Any] = {}
    for file_type, pk_cols in PRIMARY_KEYS.items():
        if file_type in dataframes:
            pk_results[file_type] = validate_primary_keys(
                dataframes[file_type], pk_cols, file_type
            )

    # 2. Validate foreign key relationships
    fk_results: list[dict[str, Any]] = []
    for rel in KNOWN_RELATIONSHIPS:
        parent_type = rel["parent"]["file_type"]
        child_type = rel["child"]["file_type"]

        if parent_type in dataframes and child_type in dataframes:
            fk_result = validate_referential_integrity(
                parent_df=dataframes[parent_type],
                child_df=dataframes[child_type],
                parent_key=rel["parent"]["key"],
                child_key=rel["child"]["key"],
                relationship_name=rel["name"],
            )
            fk_result["cardinality"] = rel["cardinality"]
            fk_result["description"] = rel["description"]
            fk_results.append(fk_result)
        else:
            missing = []
            if parent_type not in dataframes:
                missing.append(parent_type)
            if child_type not in dataframes:
                missing.append(child_type)
            fk_results.append({
                "relationship": rel["name"],
                "skipped": True,
                "reason": f"Missing data files: {missing}",
            })

    # 3. Compute additional join statistics
    join_stats = _compute_join_statistics(dataframes)

    # 4. Generate Mermaid ERD
    erd_mermaid = generate_erd_mermaid(dataframes, fk_results)

    # Build report
    report = {
        "city": city_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files_loaded": list(dataframes.keys()),
        "primary_key_validation": pk_results,
        "foreign_key_validation": fk_results,
        "join_statistics": join_stats,
        "erd_mermaid": erd_mermaid,
    }

    # Save report
    _save_relationship_report(city_name, report)

    return report


# ===================================================================
# Join statistics
# ===================================================================

def _compute_join_statistics(
    dataframes: dict[str, pl.DataFrame],
) -> dict[str, Any]:
    """Compute cross-file join statistics.

    Args:
        dataframes: Dict of file_type → DataFrame.

    Returns:
        Join statistics dict.
    """
    stats: dict[str, Any] = {}

    # Listings ↔ Calendar: rows per listing
    if "listings" in dataframes and "calendar" in dataframes:
        listings_df = dataframes["listings"]
        calendar_df = dataframes["calendar"]

        if "id" in listings_df.columns and "listing_id" in calendar_df.columns:
            calendar_per_listing = (
                calendar_df.group_by("listing_id")
                .len()
                .rename({"len": "calendar_rows"})
            )
            stats["calendar_per_listing"] = {
                "mean": round(float(calendar_per_listing["calendar_rows"].mean()), 1),
                "median": float(calendar_per_listing["calendar_rows"].median()),
                "min": int(calendar_per_listing["calendar_rows"].min()),
                "max": int(calendar_per_listing["calendar_rows"].max()),
            }

    # Listings ↔ Reviews: reviews per listing
    if "listings" in dataframes and "reviews" in dataframes:
        reviews_df = dataframes["reviews"]
        if "listing_id" in reviews_df.columns:
            reviews_per_listing = (
                reviews_df.group_by("listing_id")
                .len()
                .rename({"len": "review_count"})
            )
            stats["reviews_per_listing"] = {
                "mean": round(float(reviews_per_listing["review_count"].mean()), 1),
                "median": float(reviews_per_listing["review_count"].median()),
                "min": int(reviews_per_listing["review_count"].min()),
                "max": int(reviews_per_listing["review_count"].max()),
            }

    # Host analysis (from listings — hosts are denormalized)
    if "listings" in dataframes and "host_id" in dataframes["listings"].columns:
        listings_per_host = (
            dataframes["listings"]
            .group_by("host_id")
            .len()
            .rename({"len": "listing_count"})
        )
        stats["listings_per_host"] = {
            "mean": round(float(listings_per_host["listing_count"].mean()), 1),
            "median": float(listings_per_host["listing_count"].median()),
            "min": int(listings_per_host["listing_count"].min()),
            "max": int(listings_per_host["listing_count"].max()),
            "total_hosts": listings_per_host.height,
            "multi_listing_hosts": int(
                listings_per_host.filter(pl.col("listing_count") > 1).height
            ),
        }

    return stats


# ===================================================================
# ERD generation (Mermaid)
# ===================================================================

def generate_erd_mermaid(
    dataframes: dict[str, pl.DataFrame],
    fk_results: list[dict[str, Any]],
) -> str:
    """Generate a Mermaid ERD diagram string from loaded data and FK results.

    Args:
        dataframes: Dict of file_type → DataFrame.
        fk_results: Foreign key validation results.

    Returns:
        Mermaid ERD markdown string.
    """
    lines = ["erDiagram"]

    # Add relationships
    for fk in fk_results:
        if fk.get("skipped"):
            continue
        rel_name = fk.get("relationship", "")
        desc = fk.get("description", rel_name)
        coverage = fk.get("coverage_pct", "?")

        # Determine parent/child entity names
        for known_rel in KNOWN_RELATIONSHIPS:
            if known_rel["name"] == rel_name:
                parent = known_rel["parent"]["file_type"].upper()
                child = known_rel["child"]["file_type"].upper()
                card = known_rel["cardinality"]

                # Map cardinality to Mermaid notation
                if card == "1:N":
                    lines.append(
                        f'    {parent} ||--o{{ {child} : "{desc} ({coverage}% coverage)"'
                    )
                break

    # Add entity definitions with key columns
    for file_type, df in dataframes.items():
        entity_name = file_type.upper()
        lines.append(f"    {entity_name} {{")

        pk_cols = PRIMARY_KEYS.get(file_type, [])
        for col_name in df.columns[:15]:  # Limit to first 15 cols for readability
            dtype = str(df[col_name].dtype).lower()
            key_marker = "PK" if col_name in pk_cols else ""
            # Check if it's a foreign key
            for rel in KNOWN_RELATIONSHIPS:
                if (rel["child"]["file_type"] == file_type
                        and rel["child"]["key"] == col_name):
                    key_marker = "FK"
                    break
            lines.append(f'        {dtype} {col_name} {key_marker}'.rstrip())
        lines.append("    }")

    return "\n".join(lines)


# ===================================================================
# Output persistence
# ===================================================================

def _save_relationship_report(city_name: str, report: dict) -> Path:
    """Save relationship report to outputs/relationships/."""
    output_dir = get_output_dir("relationships")
    output_path = output_dir / f"{city_name}_relationship_report.json"

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    logger.info("Relationship report saved: %s", output_path)

    # Also save the Mermaid ERD as a separate markdown file
    erd_path = output_dir / f"{city_name}_erd.md"
    with open(erd_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Entity-Relationship Diagram: {city_name}\n\n")
        fh.write("```mermaid\n")
        fh.write(report.get("erd_mermaid", ""))
        fh.write("\n```\n")

    logger.info("ERD saved: %s", erd_path)
    return output_path
