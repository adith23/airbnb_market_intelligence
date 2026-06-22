"""Data enrichment and joining pipeline for Section 3.3.

This module turns cleaned staging Parquet files into analytics-ready
gold-layer listing datasets. It keeps listings as the base grain, then
left-joins optional calendar, review, neighbourhood, city, and currency
context onto that base.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pipeline.utils import (
    CONFIG_DIR,
    get_enriched_dir,
    get_output_dir,
    get_staging_dir,
    load_city_config,
    load_yaml_config,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrichmentResult:
    """Summary of a single-city enrichment run."""

    city: str
    listings_count: int
    calendar_rows_aggregated: int
    reviews_aggregated: int
    derived_fields_added: list[str]
    join_coverage: dict[str, float]
    output_path: str
    warnings: list[str] = field(default_factory=list)


def _load_optional_parquet(path: Path) -> pl.DataFrame | None:
    """Read a Parquet file if it exists."""
    if not path.exists():
        logger.warning("Optional staging file not found: %s", path)
        return None
    return pl.read_parquet(path)


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first candidate column present in a column list."""
    available = set(columns)
    return next((col for col in candidates if col in available), None)


def _listing_id_column(df: pl.DataFrame) -> str:
    """Resolve the listing natural-key column from known variants."""
    col = _first_existing(df.columns, ["listing_id", "id"])
    if col is None:
        raise ValueError("Listings data must contain either 'listing_id' or 'id'.")
    return col


def _price_column(df: pl.DataFrame) -> str | None:
    """Resolve local price column from current and canonical names."""
    return _first_existing(df.columns, ["price_local", "price"])


def _safe_date(value: str | None) -> date | None:
    """Parse a config date without leaking low-level errors."""
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        logger.warning("Invalid scrape_date in city config: %s", value)
        return None


def _active_listing_config() -> tuple[int, int]:
    """Load active-listing thresholds from enrichment config."""
    try:
        config = load_yaml_config(CONFIG_DIR / "enrichment_config.yaml")
    except Exception:
        logger.warning("Could not load enrichment config; using active-listing defaults")
        return 12, 1

    active = config.get("active_listing", {})
    return (
        int(active.get("review_recency_months", 12)),
        int(active.get("min_availability_365", 1)),
    )


def _professional_host_threshold() -> int:
    """Load the professional-host threshold from enrichment config."""
    try:
        config = load_yaml_config(CONFIG_DIR / "enrichment_config.yaml")
        return int(config.get("professional_host", {}).get("min_listings", 3))
    except Exception:
        logger.warning("Could not load professional-host threshold; using 3")
        return 3


def _load_exchange_rates() -> dict[str, float]:
    """Load currency conversion rates to USD."""
    try:
        config = load_yaml_config(CONFIG_DIR / "enrichment_config.yaml")
    except Exception:
        logger.warning("Could not load exchange rates; using USD=1.0")
        return {"USD": 1.0}

    rates = config.get("exchange_rates_to_usd", {"USD": 1.0})
    return {str(currency): float(rate) for currency, rate in rates.items()}


def _aggregate_calendar(staging_dir: Path) -> pl.DataFrame | None:
    """Aggregate daily calendar records to one row per listing.

    Inside Airbnb's ``available = false`` includes both booked and
    host-blocked dates, so occupancy and revenue metrics are upper-bound
    estimates rather than audited financial actuals.
    """
    df = _load_optional_parquet(staging_dir / "calendar.parquet")
    if df is None:
        return None

    required = {"listing_id", "available"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Calendar staging data is missing required columns: {sorted(missing)}")

    price_col = _price_column(df)

    agg_exprs: list[pl.Expr] = [
        pl.len().alias("total_days"),
        pl.col("available").cast(pl.UInt8).sum().cast(pl.Int64).alias("days_available"),
        (~pl.col("available")).cast(pl.UInt8).sum().cast(pl.Int64).alias("days_booked"),
    ]

    if price_col is not None:
        agg_exprs.extend([
            pl.col(price_col).filter(pl.col("available")).mean().alias("avg_asking_price"),
            pl.col(price_col).filter(pl.col("available")).median().alias("median_asking_price"),
            pl.col(price_col).filter(~pl.col("available")).sum().alias("estimated_annual_revenue"),
            pl.col(price_col).filter(~pl.col("available")).mean().alias("avg_booked_price"),
        ])

    if "adjusted_price" in df.columns:
        agg_exprs.append(
            pl.col("adjusted_price")
            .filter(pl.col("available"))
            .mean()
            .alias("avg_adjusted_asking_price")
        )
    elif "adjusted_price_local" in df.columns:
        agg_exprs.append(
            pl.col("adjusted_price_local")
            .filter(pl.col("available"))
            .mean()
            .alias("avg_adjusted_asking_price")
        )

    if "minimum_nights" in df.columns:
        agg_exprs.append(pl.col("minimum_nights").mean().alias("avg_min_nights_cal"))
    if "maximum_nights" in df.columns:
        agg_exprs.append(pl.col("maximum_nights").mean().alias("avg_max_nights_cal"))

    agg = df.group_by("listing_id").agg(agg_exprs)
    agg = agg.with_columns(
        pl.when(pl.col("total_days") > 0)
        .then((pl.col("days_booked") / pl.col("total_days") * 100).round(2))
        .otherwise(None)
        .alias("occupancy_rate_pct")
    )
    if price_col is not None:
        agg = agg.with_columns(
            (pl.col("occupancy_rate_pct") / 100.0 * pl.col("avg_booked_price") * 30)
            .round(2)
            .alias("estimated_monthly_revenue")
        )

    logger.info("Calendar aggregated: %d rows -> %d listings", df.height, agg.height)
    return agg


def _aggregate_reviews(staging_dir: Path) -> pl.DataFrame | None:
    """Aggregate review events to one row per listing."""
    df = _load_optional_parquet(staging_dir / "reviews.parquet")
    if df is None:
        return None

    required = {"listing_id", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Reviews staging data is missing required columns: {sorted(missing)}")

    agg_exprs: list[pl.Expr] = [
        pl.len().alias("review_count_computed"),
        pl.col("date").min().alias("first_review_date_computed"),
        pl.col("date").max().alias("last_review_date_computed"),
    ]
    if "reviewer_id" in df.columns:
        agg_exprs.append(pl.col("reviewer_id").n_unique().alias("unique_reviewers"))
    if "comments" in df.columns:
        agg_exprs.extend(
            [
                pl.col("comments").str.len_chars().mean().round(0).alias("avg_comment_length"),
                pl.col("comments").str.len_chars().median().alias("median_comment_length"),
            ]
        )

    agg = df.group_by("listing_id").agg(agg_exprs)
    logger.info("Reviews aggregated: %d rows -> %d listings", df.height, agg.height)
    return agg


def _load_neighbourhood_reference(staging_dir: Path) -> pl.DataFrame | None:
    """Load cleaned neighbourhood reference data, if present."""
    df = _load_optional_parquet(staging_dir / "neighbourhoods.parquet")
    if df is None or df.is_empty():
        return None

    rename_map = {
        "neighbourhood": "neighbourhood_name",
        "neighborhood": "neighbourhood_name",
        "neighbourhood_group": "neighbourhood_group",
        "neighborhood_group": "neighbourhood_group",
    }
    applicable = {src: dst for src, dst in rename_map.items() if src in df.columns and src != dst}
    if applicable:
        df = df.rename(applicable)

    if "neighbourhood_name" not in df.columns:
        return None

    select_cols = ["neighbourhood_name"]
    if "neighbourhood_group" in df.columns:
        select_cols.append("neighbourhood_group")
    return df.select(select_cols).unique()


def _aggregate_neighbourhoods(
    listings_df: pl.DataFrame,
    neighbourhood_ref: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute neighbourhood benchmarks and attach reference attributes."""
    group_col = _first_existing(
        listings_df.columns,
        ["neighbourhood_cleansed", "neighbourhood_name", "neighbourhood"],
    )
    if group_col is None:
        logger.warning("No neighbourhood column found; skipping neighbourhood aggregates")
        return pl.DataFrame()

    price_col = _price_column(listings_df)
    agg_exprs: list[pl.Expr] = [pl.len().alias("neighbourhood_listing_count")]

    if price_col is not None:
        agg_exprs.extend(
            [
                pl.col(price_col).median().alias("neighbourhood_median_price"),
                pl.col(price_col).mean().round(2).alias("neighbourhood_mean_price"),
            ]
        )
    if "review_scores_rating" in listings_df.columns:
        agg_exprs.append(
            pl.col("review_scores_rating").mean().round(2).alias("neighbourhood_avg_rating")
        )
    if "availability_365" in listings_df.columns:
        agg_exprs.append(
            pl.col("availability_365").mean().round(1).alias("neighbourhood_avg_availability")
        )

    agg = listings_df.group_by(group_col).agg(agg_exprs)
    if group_col != "neighbourhood_cleansed":
        agg = agg.rename({group_col: "neighbourhood_cleansed"})

    if neighbourhood_ref is not None and "neighbourhood_name" in neighbourhood_ref.columns:
        agg = agg.join(
            neighbourhood_ref,
            left_on="neighbourhood_cleansed",
            right_on="neighbourhood_name",
            how="left",
        )

    logger.info("Neighbourhood aggregates created: %d rows", agg.height)
    return agg


def _join_master(
    listings_df: pl.DataFrame,
    calendar_agg: pl.DataFrame | None,
    review_agg: pl.DataFrame | None,
    neighbourhood_agg: pl.DataFrame | None,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Assemble the listing-grain master table with left joins."""
    master = listings_df
    join_key = _listing_id_column(master)
    total = max(master.height, 1)
    coverage: dict[str, float] = {}

    if calendar_agg is not None and not calendar_agg.is_empty():
        master = master.join(calendar_agg, left_on=join_key, right_on="listing_id", how="left")
        matched = master.filter(pl.col("total_days").is_not_null()).height
        coverage["calendar"] = round(matched / total * 100, 1)

    if review_agg is not None and not review_agg.is_empty():
        master = master.join(review_agg, left_on=join_key, right_on="listing_id", how="left")
        matched = master.filter(pl.col("review_count_computed").is_not_null()).height
        coverage["reviews"] = round(matched / total * 100, 1)

    if neighbourhood_agg is not None and not neighbourhood_agg.is_empty():
        nbhd_col = _first_existing(master.columns, ["neighbourhood_cleansed", "neighbourhood"])
        if nbhd_col is not None and "neighbourhood_cleansed" in neighbourhood_agg.columns:
            master = master.join(
                neighbourhood_agg,
                left_on=nbhd_col,
                right_on="neighbourhood_cleansed",
                how="left",
            )
            matched = master.filter(pl.col("neighbourhood_listing_count").is_not_null()).height
            coverage["neighbourhood"] = round(matched / total * 100, 1)

    return master, coverage


def _compute_derived_fields(df: pl.DataFrame, scrape_date: date | None) -> tuple[pl.DataFrame, list[str]]:
    """Add vectorized business metrics to the master listings table."""
    fields_added: list[str] = []
    price_col = _price_column(df)

    if price_col == "price" and "price_local" not in df.columns:
        df = df.with_columns(pl.col("price").alias("price_local"))
        fields_added.append("price_local")
        price_col = "price_local"

    if "host_since" in df.columns and scrape_date is not None:
        df = df.with_columns(
            ((pl.lit(scrape_date).cast(pl.Date) - pl.col("host_since")).dt.total_days() / 365.25)
            .round(2)
            .alias("host_tenure_years")
        )
        fields_added.append("host_tenure_years")

    if "number_of_reviews" in df.columns and "first_review" in df.columns and scrape_date:
        months_since = (
            (pl.lit(scrape_date).cast(pl.Date) - pl.col("first_review")).dt.total_days() / 30.44
        ).clip(lower_bound=1)
        df = df.with_columns(
            pl.when(pl.col("first_review").is_not_null() & (pl.col("number_of_reviews") > 0))
            .then((pl.col("number_of_reviews") / months_since).round(2))
            .otherwise(None)
            .alias("review_frequency_monthly")
        )
        fields_added.append("review_frequency_monthly")

    if price_col is not None and "bedrooms" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("bedrooms").is_not_null() & (pl.col("bedrooms") > 0))
            .then((pl.col(price_col) / pl.col("bedrooms")).round(2))
            .otherwise(None)
            .alias("price_per_bedroom")
        )
        fields_added.append("price_per_bedroom")

    if price_col is not None and "accommodates" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("accommodates").is_not_null() & (pl.col("accommodates") > 0))
            .then((pl.col(price_col) / pl.col("accommodates")).round(2))
            .otherwise(None)
            .alias("price_per_person")
        )
        fields_added.append("price_per_person")

    if "last_review" in df.columns and "availability_365" in df.columns and scrape_date:
        review_months, min_availability = _active_listing_config()
        cutoff = scrape_date - timedelta(days=round(review_months * 30.44))
        df = df.with_columns(
            (
                (pl.col("last_review").is_not_null() & (pl.col("last_review") >= cutoff))
                | (
                    pl.col("availability_365").is_not_null()
                    & (pl.col("availability_365") >= min_availability)
                )
            ).alias("is_active")
        )
        fields_added.append("is_active")

    host_count_col = _first_existing(
        df.columns,
        ["calculated_host_listings_count", "host_listings_count", "host_total_listings_count"],
    )
    if host_count_col is not None:
        threshold = _professional_host_threshold()
        df = df.with_columns(
            (pl.col(host_count_col).is_not_null() & (pl.col(host_count_col) >= threshold)).alias(
                "is_professional_host"
            )
        )
        fields_added.append("is_professional_host")

    if "estimated_monthly_revenue" not in df.columns and "occupancy_rate_pct" in df.columns:
        revenue_base_price = "avg_booked_price" if "avg_booked_price" in df.columns else price_col
        if revenue_base_price and revenue_base_price in df.columns:
            df = df.with_columns(
                (pl.col("occupancy_rate_pct") / 100.0 * pl.col(revenue_base_price) * 30)
                .round(2)
                .alias("estimated_monthly_revenue")
            )
            fields_added.append("estimated_monthly_revenue")

    if "estimated_annual_revenue" not in df.columns and "estimated_monthly_revenue" in df.columns:
        df = df.with_columns(
            (pl.col("estimated_monthly_revenue") * 12).round(2).alias("estimated_annual_revenue")
        )
        fields_added.append("estimated_annual_revenue")

    if price_col is not None and "neighbourhood_median_price" in df.columns:
        df = df.with_columns(
            pl.when(
                pl.col("neighbourhood_median_price").is_not_null()
                & (pl.col("neighbourhood_median_price") > 0)
            )
            .then((pl.col(price_col) / pl.col("neighbourhood_median_price")).round(3))
            .otherwise(None)
            .alias("price_vs_neighbourhood")
        )
        fields_added.append("price_vs_neighbourhood")

    return df, fields_added


def _add_city_metadata(df: pl.DataFrame, city_name: str, city_config: dict[str, Any]) -> pl.DataFrame:
    """Add city/snapshot context needed for cross-city analytics."""
    scrape_date = _safe_date(city_config.get("scrape_date"))
    return df.with_columns(
        [
            pl.lit(city_name).alias("city"),
            pl.lit(city_config.get("display_name", city_name)).alias("city_display_name"),
            pl.lit(city_config.get("country", "")).alias("country"),
            pl.lit(city_config.get("currency_code", "USD")).alias("currency_code"),
            pl.lit(city_config.get("currency_symbol", "")).alias("currency_symbol"),
            pl.lit(city_config.get("timezone", "")).alias("timezone"),
            pl.lit(scrape_date).cast(pl.Date).alias("scrape_date"),
        ]
    )


def _add_usd_prices(
    df: pl.DataFrame,
    currency_code: str,
    exchange_rates: dict[str, float],
) -> pl.DataFrame:
    """Add USD-normalized price and revenue measures."""
    rate = float(exchange_rates.get(currency_code, 1.0))
    price_columns = [
        "price_local",
        "avg_asking_price",
        "median_asking_price",
        "avg_booked_price",
        "estimated_monthly_revenue",
        "estimated_annual_revenue",
        "neighbourhood_median_price",
        "neighbourhood_mean_price",
    ]

    exprs: list[pl.Expr] = []
    for col in price_columns:
        if col in df.columns:
            target = "price_usd" if col == "price_local" else f"{col}_usd"
            exprs.append((pl.col(col) * rate).round(2).alias(target))

    return df.with_columns(exprs) if exprs else df


def enrich_city(city_name: str) -> EnrichmentResult:
    """Build ``data/enriched/{city}_master_listings.parquet`` from staging files."""
    staging_dir = get_staging_dir(city_name)
    listings_path = staging_dir / "listings.parquet"
    if not listings_path.exists():
        raise FileNotFoundError(
            f"Staging listings not found: {listings_path}. "
            f"Run cleaning first: python main.py clean --city {city_name}"
        )

    city_config = load_city_config(city_name)
    scrape_date = _safe_date(city_config.get("scrape_date"))

    logger.info("Starting enrichment for city=%s", city_name)
    listings_df = pl.read_parquet(listings_path)
    _listing_id_column(listings_df)

    neighbourhood_ref = _load_neighbourhood_reference(staging_dir)
    calendar_agg = _aggregate_calendar(staging_dir)
    review_agg = _aggregate_reviews(staging_dir)
    neighbourhood_agg = _aggregate_neighbourhoods(listings_df, neighbourhood_ref)

    master, coverage = _join_master(listings_df, calendar_agg, review_agg, neighbourhood_agg)
    master, fields_added = _compute_derived_fields(master, scrape_date)
    master = _add_city_metadata(master, city_name, city_config)
    master = _add_usd_prices(
        master,
        str(city_config.get("currency_code", "USD")),
        _load_exchange_rates(),
    )

    enriched_dir = get_enriched_dir()
    output_path = enriched_dir / f"{city_name}_master_listings.parquet"
    master.write_parquet(output_path)

    warnings = []
    if calendar_agg is None:
        warnings.append("calendar_missing")
    if review_agg is None:
        warnings.append("reviews_missing")

    _save_enrichment_report(
        result=EnrichmentResult(
            city=city_name,
            listings_count=master.height,
            calendar_rows_aggregated=calendar_agg.height if calendar_agg is not None else 0,
            reviews_aggregated=review_agg.height if review_agg is not None else 0,
            derived_fields_added=fields_added,
            join_coverage=coverage,
            output_path=str(output_path),
            warnings=warnings,
        )
    )

    logger.info("Enrichment complete: %s (%d listings)", output_path, master.height)
    return EnrichmentResult(
        city=city_name,
        listings_count=master.height,
        calendar_rows_aggregated=calendar_agg.height if calendar_agg is not None else 0,
        reviews_aggregated=review_agg.height if review_agg is not None else 0,
        derived_fields_added=fields_added,
        join_coverage=coverage,
        output_path=str(output_path),
        warnings=warnings,
    )


def _align_dataframes(dataframes: list[pl.DataFrame]) -> list[pl.DataFrame]:
    """Align DataFrames to a common union schema and stable column order."""
    schema: dict[str, pl.DataType] = {}
    for df in dataframes:
        for name, dtype in zip(df.columns, df.dtypes):
            if name not in schema:
                schema[name] = dtype
            elif schema[name] != dtype:
                if dtype in (pl.Utf8, pl.String) or schema[name] in (pl.Utf8, pl.String):
                    schema[name] = pl.String() if hasattr(pl, "String") else pl.Utf8()
                elif dtype in (pl.Float64, pl.Float32) or schema[name] in (pl.Float64, pl.Float32):
                    schema[name] = pl.Float64()
                elif dtype in (pl.Int64, pl.Int32) or schema[name] in (pl.Int64, pl.Int32):
                    schema[name] = pl.Int64()

    ordered_columns = sorted(schema)
    aligned = []
    for df in dataframes:
        exprs = []
        for col in ordered_columns:
            if col not in df.columns:
                exprs.append(pl.lit(None).cast(schema[col]).alias(col))
            else:
                if df.schema[col] != schema[col]:
                    exprs.append(pl.col(col).cast(schema[col], strict=False).alias(col))
                else:
                    exprs.append(pl.col(col))
        aligned.append(df.select(exprs))
    return aligned


def build_unified_master(city_names: list[str]) -> Path:
    """Build a cross-city unified master listings Parquet file."""
    enriched_dir = get_enriched_dir()
    dataframes: list[pl.DataFrame] = []
    missing: list[str] = []

    for city_name in city_names:
        path = enriched_dir / f"{city_name}_master_listings.parquet"
        if not path.exists():
            missing.append(city_name)
            continue
        dataframes.append(pl.read_parquet(path))

    if missing:
        logger.warning("Missing enriched data for cities: %s", ", ".join(missing))
    if len(dataframes) < 2:
        raise ValueError(
            f"Need enriched data for at least 2 cities. Found {len(dataframes)}. "
            "Run python main.py enrich --city <city> for each city first."
        )

    unified = pl.concat(_align_dataframes(dataframes), how="vertical")
    output_path = enriched_dir / "unified_master_listings.parquet"
    unified.write_parquet(output_path)
    logger.info("Unified master written: %s (%d rows)", output_path, unified.height)
    return output_path


def _save_enrichment_report(result: EnrichmentResult) -> Path:
    """Persist an enrichment run report for audit and debugging."""
    output_dir = get_output_dir("enrichment")
    output_path = output_dir / f"{result.city}_enrichment_report.json"
    report = asdict(result)
    report["enriched_at"] = datetime.now(timezone.utc).isoformat()

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    logger.info("Enrichment report saved: %s", output_path)
    return output_path
