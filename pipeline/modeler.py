"""DuckDB star-schema builder for Section 3.4.

The modeler loads enriched master listings plus cleaned staging event
tables into an analytics-ready dimensional model:

Dimensions:
  dim_date, dim_city, dim_host, dim_property, dim_neighbourhood, dim_reviewer

Facts:
  fact_listing_snapshot, fact_calendar, fact_review
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from pipeline.utils import (
    CONFIG_DIR,
    get_db_path,
    get_enriched_dir,
    get_staging_dir,
    load_city_config,
    load_yaml_config,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelingResult:
    """Summary of a star-schema build."""

    db_path: str
    cities: list[str]
    table_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)


def get_connection(db_path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection for the analytics database."""
    path = Path(db_path) if db_path is not None else get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def _quote(value: str | Path) -> str:
    """Return a SQL string literal."""
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def _column_names(con: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    """Return lowercase column names for a DuckDB table or view."""
    rows = con.execute(f"DESCRIBE {table_name}").fetchall()
    return {str(row[0]).lower() for row in rows}


def _has_table(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """Check whether a table/view exists in the current connection."""
    try:
        con.execute(f"DESCRIBE {table_name}").fetchall()
        return True
    except duckdb.CatalogException:
        return False


def _expr(
    columns: set[str],
    candidates: list[str],
    sql_type: str,
    alias: str,
    default_sql: str = "NULL",
) -> str:
    """Build a resilient select expression from candidate source columns."""
    for candidate in candidates:
        if candidate.lower() in columns:
            return f"TRY_CAST({candidate} AS {sql_type}) AS {alias}"
    return f"CAST({default_sql} AS {sql_type}) AS {alias}"


def _raw_expr(columns: set[str], candidates: list[str], sql_type: str, default_sql: str = "NULL") -> str:
    """Build a resilient unaliased expression."""
    for candidate in candidates:
        if candidate.lower() in columns:
            return f"TRY_CAST({candidate} AS {sql_type})"
    return f"CAST({default_sql} AS {sql_type})"


def _coalesce_expr(
    columns: set[str],
    candidates: list[str],
    sql_type: str,
    alias: str,
    default_sql: str = "NULL",
) -> str:
    """Build a COALESCE expression over available candidate columns."""
    available = [f"TRY_CAST({candidate} AS {sql_type})" for candidate in candidates if candidate.lower() in columns]
    if not available:
        return f"CAST({default_sql} AS {sql_type}) AS {alias}"
    return f"COALESCE({', '.join(available)}) AS {alias}"


def _city_usd_rate_case() -> str:
    """Build a DuckDB CASE expression mapping city name to USD rate."""
    try:
        rates = load_yaml_config(CONFIG_DIR / "enrichment_config.yaml").get(
            "exchange_rates_to_usd",
            {"USD": 1.0},
        )
    except Exception:
        rates = {"USD": 1.0}

    cities = load_city_config()
    cases: list[str] = []
    for city_name, city_config in cities.items():
        currency = str(city_config.get("currency_code", "USD"))
        rate = float(rates.get(currency, 1.0))
        cases.append(f"WHEN {_quote(city_name)} THEN {rate}")
    return "CASE c.city " + " ".join(cases) + " ELSE 1.0 END"


def _stage_required_enriched(city_names: list[str]) -> list[Path]:
    """Resolve required enriched master files for model input."""
    enriched_dir = get_enriched_dir()
    paths: list[Path] = []
    missing: list[str] = []
    for city_name in city_names:
        path = enriched_dir / f"{city_name}_master_listings.parquet"
        if path.exists():
            paths.append(path)
        else:
            missing.append(city_name)
    if missing:
        raise FileNotFoundError(
            "Missing enriched master files for cities: "
            + ", ".join(missing)
            + ". Run python main.py enrich --city <city> first."
        )
    return paths


def _existing_staging_files(city_names: list[str], file_type: str) -> list[Path]:
    """Return existing cleaned staging files for a logical file type."""
    return [
        get_staging_dir(city_name) / f"{file_type}.parquet"
        for city_name in city_names
        if (get_staging_dir(city_name) / f"{file_type}.parquet").exists()
    ]


def _create_parquet_view(
    con: duckdb.DuckDBPyConnection,
    view_name: str,
    paths: list[Path],
    add_city_from_path: bool = False,
) -> None:
    """Create a temp view over one or more Parquet files."""
    if not paths:
        return

    path_list = ", ".join(_quote(path) for path in paths)
    source = f"read_parquet([{path_list}], union_by_name=true, filename=true)"
    if add_city_from_path:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW {view_name} AS
            SELECT
                regexp_extract(filename, 'staging/([^/]+)/[^/]+\\.parquet', 1) AS city,
                * EXCLUDE (filename)
            FROM {source}
            """
        )
    else:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW {view_name} AS
            SELECT * EXCLUDE (filename)
            FROM {source}
            """
        )


def _create_dim_city(con: duckdb.DuckDBPyConnection, city_names: list[str]) -> None:
    """Load city dimension from config/cities.yaml."""
    rows: list[str] = []
    for idx, city_name in enumerate(city_names, start=1):
        config = load_city_config(city_name)
        rows.append(
            "("
            f"{idx}, "
            f"{_quote(city_name)}, "
            f"{_quote(config.get('display_name', city_name))}, "
            f"{_quote(config.get('country', ''))}, "
            f"{_quote(config.get('currency_code', 'USD'))}, "
            f"{_quote(config.get('currency_symbol', ''))}, "
            f"{_quote(config.get('timezone', ''))}"
            ")"
        )

    values_sql = ", ".join(rows)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE dim_city AS
        SELECT
            city_key::INTEGER AS city_key,
            city_name::VARCHAR AS city_name,
            display_name::VARCHAR AS display_name,
            country::VARCHAR AS country,
            currency_code::VARCHAR AS currency_code,
            currency_symbol::VARCHAR AS currency_symbol,
            timezone::VARCHAR AS timezone
        FROM (
            VALUES {values_sql}
        ) AS t(city_key, city_name, display_name, country, currency_code, currency_symbol, timezone)
        """
    )


def _create_dim_host(con: duckdb.DuckDBPyConnection) -> None:
    """Create one host dimension row per host_id."""
    cols = _column_names(con, "stg_master")
    select_fields = [
        _expr(cols, ["host_id"], "BIGINT", "host_id"),
        _expr(cols, ["host_name"], "VARCHAR", "host_name"),
        _expr(cols, ["host_since"], "DATE", "host_since"),
        _expr(cols, ["host_location"], "VARCHAR", "host_location"),
        _expr(cols, ["host_response_time"], "VARCHAR", "host_response_time"),
        _expr(cols, ["host_response_rate"], "DOUBLE", "host_response_rate"),
        _expr(cols, ["host_acceptance_rate"], "DOUBLE", "host_acceptance_rate"),
        _expr(cols, ["host_is_superhost"], "BOOLEAN", "host_is_superhost"),
        _coalesce_expr(cols, ["host_listings_count", "calculated_host_listings_count"], "INTEGER", "host_listings_count"),
        _expr(cols, ["host_total_listings_count"], "INTEGER", "host_total_listings_count"),
        _expr(cols, ["host_has_profile_pic"], "BOOLEAN", "host_has_profile_pic"),
        _expr(cols, ["host_identity_verified"], "BOOLEAN", "host_identity_verified"),
        _expr(cols, ["host_verification_count"], "INTEGER", "host_verification_count"),
        _expr(cols, ["is_professional_host"], "BOOLEAN", "is_professional_host"),
        _expr(cols, ["host_tenure_years"], "DOUBLE", "host_tenure_years"),
    ]

    con.execute(
        f"""
        CREATE OR REPLACE TABLE dim_host AS
        WITH host_base AS (
            SELECT {", ".join(select_fields)}
            FROM stg_master
            WHERE host_id IS NOT NULL
        ),
        deduped AS (
            SELECT
                host_id,
                any_value(host_name) AS host_name,
                min(host_since) AS host_since,
                any_value(host_location) AS host_location,
                any_value(host_response_time) AS host_response_time,
                max(host_response_rate) AS host_response_rate,
                max(host_acceptance_rate) AS host_acceptance_rate,
                bool_or(coalesce(host_is_superhost, false)) AS host_is_superhost,
                max(host_listings_count) AS host_listings_count,
                max(host_total_listings_count) AS host_total_listings_count,
                bool_or(coalesce(host_has_profile_pic, false)) AS host_has_profile_pic,
                bool_or(coalesce(host_identity_verified, false)) AS host_identity_verified,
                max(host_verification_count) AS host_verification_count,
                bool_or(coalesce(is_professional_host, false)) AS is_professional_host,
                max(host_tenure_years) AS host_tenure_years
            FROM host_base
            GROUP BY host_id
        )
        SELECT row_number() OVER (ORDER BY host_id) AS host_key, *
        FROM deduped
        """
    )


def _create_dim_property(con: duckdb.DuckDBPyConnection) -> None:
    """Create one property dimension row per listing_id."""
    cols = _column_names(con, "stg_master")
    listing_id = _coalesce_expr(cols, ["listing_id", "id"], "BIGINT", "listing_id")
    fields = [
        listing_id,
        _expr(cols, ["listing_url"], "VARCHAR", "listing_url"),
        _expr(cols, ["name"], "VARCHAR", "name"),
        _expr(cols, ["description"], "VARCHAR", "description"),
        _expr(cols, ["property_type"], "VARCHAR", "property_type"),
        _expr(cols, ["room_type"], "VARCHAR", "room_type"),
        _expr(cols, ["accommodates"], "INTEGER", "accommodates"),
        _expr(cols, ["bathrooms"], "DOUBLE", "bathrooms"),
        _expr(cols, ["bathrooms_shared"], "BOOLEAN", "bathrooms_shared"),
        _expr(cols, ["bedrooms"], "INTEGER", "bedrooms"),
        _expr(cols, ["beds"], "INTEGER", "beds"),
        _expr(cols, ["amenities"], "VARCHAR", "amenities"),
        _expr(cols, ["amenity_count"], "INTEGER", "amenity_count"),
        _expr(cols, ["instant_bookable"], "BOOLEAN", "instant_bookable"),
        _expr(cols, ["license", "licence"], "VARCHAR", "license"),
    ]
    con.execute(
        f"""
        CREATE OR REPLACE TABLE dim_property AS
        WITH property_base AS (
            SELECT {", ".join(fields)}
            FROM stg_master
        ),
        deduped AS (
            SELECT
                listing_id,
                any_value(listing_url) AS listing_url,
                any_value(name) AS name,
                any_value(description) AS description,
                any_value(property_type) AS property_type,
                any_value(room_type) AS room_type,
                max(accommodates) AS accommodates,
                max(bathrooms) AS bathrooms,
                bool_or(coalesce(bathrooms_shared, false)) AS bathrooms_shared,
                max(bedrooms) AS bedrooms,
                max(beds) AS beds,
                any_value(amenities) AS amenities,
                max(amenity_count) AS amenity_count,
                bool_or(coalesce(instant_bookable, false)) AS instant_bookable,
                any_value(license) AS license
            FROM property_base
            WHERE listing_id IS NOT NULL
            GROUP BY listing_id
        )
        SELECT row_number() OVER (ORDER BY listing_id) AS property_key, *
        FROM deduped
        """
    )


def _create_dim_neighbourhood(con: duckdb.DuckDBPyConnection) -> None:
    """Create one neighbourhood dimension row per city/neighbourhood."""
    cols = _column_names(con, "stg_master")
    fields = [
        _expr(cols, ["city"], "VARCHAR", "city"),
        _coalesce_expr(cols, ["neighbourhood_cleansed", "neighbourhood_name", "neighbourhood"], "VARCHAR", "neighbourhood_name"),
        _coalesce_expr(cols, ["neighbourhood_group", "neighbourhood_group_cleansed"], "VARCHAR", "neighbourhood_group"),
        _expr(cols, ["neighbourhood_listing_count"], "INTEGER", "listing_count"),
        _expr(cols, ["neighbourhood_median_price"], "DOUBLE", "median_price"),
        _expr(cols, ["neighbourhood_mean_price"], "DOUBLE", "mean_price"),
        _expr(cols, ["neighbourhood_avg_rating"], "DOUBLE", "avg_rating"),
        _expr(cols, ["neighbourhood_avg_availability"], "DOUBLE", "avg_availability"),
    ]
    con.execute(
        f"""
        CREATE OR REPLACE TABLE dim_neighbourhood AS
        WITH neighbourhood_base AS (
            SELECT {", ".join(fields)}
            FROM stg_master
        ),
        deduped AS (
            SELECT
                city,
                neighbourhood_name,
                any_value(neighbourhood_group) AS neighbourhood_group,
                max(listing_count) AS listing_count,
                max(median_price) AS median_price,
                max(mean_price) AS mean_price,
                max(avg_rating) AS avg_rating,
                max(avg_availability) AS avg_availability
            FROM neighbourhood_base
            WHERE city IS NOT NULL AND neighbourhood_name IS NOT NULL
            GROUP BY city, neighbourhood_name
        )
        SELECT row_number() OVER (ORDER BY city, neighbourhood_name) AS neighbourhood_key, *
        FROM deduped
        """
    )


def _create_dim_reviewer(con: duckdb.DuckDBPyConnection) -> None:
    """Create reviewer dimension from staged reviews."""
    if not _has_table(con, "stg_reviews"):
        con.execute(
            """
            CREATE OR REPLACE TABLE dim_reviewer (
                reviewer_key INTEGER,
                reviewer_id BIGINT,
                reviewer_name VARCHAR
            )
            """
        )
        return

    cols = _column_names(con, "stg_reviews")
    fields = [
        _expr(cols, ["reviewer_id"], "BIGINT", "reviewer_id"),
        _expr(cols, ["reviewer_name"], "VARCHAR", "reviewer_name"),
    ]
    con.execute(
        f"""
        CREATE OR REPLACE TABLE dim_reviewer AS
        WITH reviewer_base AS (
            SELECT {", ".join(fields)}
            FROM stg_reviews
        ),
        deduped AS (
            SELECT reviewer_id, any_value(reviewer_name) AS reviewer_name
            FROM reviewer_base
            WHERE reviewer_id IS NOT NULL
            GROUP BY reviewer_id
        )
        SELECT row_number() OVER (ORDER BY reviewer_id) AS reviewer_key, *
        FROM deduped
        """
    )


def _date_source_sql(con: duckdb.DuckDBPyConnection) -> str:
    """Build a union query of all known date-bearing fields."""
    pieces = ["SELECT TRY_CAST(scrape_date AS DATE) AS d FROM stg_master"]
    if _has_table(con, "stg_calendar"):
        pieces.append("SELECT TRY_CAST(date AS DATE) AS d FROM stg_calendar")
    if _has_table(con, "stg_reviews"):
        pieces.append("SELECT TRY_CAST(date AS DATE) AS d FROM stg_reviews")
    return " UNION ALL ".join(pieces)


def _create_dim_date(con: duckdb.DuckDBPyConnection) -> None:
    """Generate a dense date dimension covering all model dates."""
    min_date, max_date = con.execute(
        f"SELECT min(d), max(d) FROM ({_date_source_sql(con)}) WHERE d IS NOT NULL"
    ).fetchone()

    if min_date is None or max_date is None:
        con.execute(
            """
            CREATE OR REPLACE TABLE dim_date (
                date_key INTEGER,
                full_date DATE,
                year SMALLINT,
                quarter TINYINT,
                month TINYINT,
                day_of_month TINYINT,
                day_of_week TINYINT,
                day_name VARCHAR,
                month_name VARCHAR,
                is_weekend BOOLEAN,
                week_of_year TINYINT
            )
            """
        )
        return

    con.execute(
        f"""
        CREATE OR REPLACE TABLE dim_date AS
        SELECT
            CAST(strftime(full_date, '%Y%m%d') AS INTEGER) AS date_key,
            full_date,
            CAST(year(full_date) AS SMALLINT) AS year,
            CAST(quarter(full_date) AS TINYINT) AS quarter,
            CAST(month(full_date) AS TINYINT) AS month,
            CAST(day(full_date) AS TINYINT) AS day_of_month,
            CAST(strftime(full_date, '%w') AS TINYINT) AS day_of_week,
            strftime(full_date, '%A') AS day_name,
            strftime(full_date, '%B') AS month_name,
            strftime(full_date, '%w') IN ('0', '6') AS is_weekend,
            CAST(strftime(full_date, '%W') AS TINYINT) AS week_of_year
        FROM generate_series(DATE {_quote(min_date)}, DATE {_quote(max_date)}, INTERVAL 1 DAY)
            AS t(full_date)
        """
    )


def _create_fact_listing_snapshot(con: duckdb.DuckDBPyConnection) -> None:
    """Create listing snapshot fact at listing x scrape-date grain."""
    cols = _column_names(con, "stg_master")
    listing_id = _raw_expr(cols, ["listing_id", "id"], "BIGINT")
    city = _raw_expr(cols, ["city"], "VARCHAR")
    neighbourhood = _raw_expr(cols, ["neighbourhood_cleansed", "neighbourhood_name", "neighbourhood"], "VARCHAR")
    fields = [
        f"{listing_id} AS listing_id",
        _expr(cols, ["host_id"], "BIGINT", "host_id"),
        f"{city} AS city",
        f"{neighbourhood} AS neighbourhood_name",
        _expr(cols, ["scrape_date"], "DATE", "snapshot_date"),
        _coalesce_expr(cols, ["price_local", "price"], "DOUBLE", "price_local"),
        _expr(cols, ["price_usd"], "DOUBLE", "price_usd"),
        _expr(cols, ["latitude"], "DOUBLE", "latitude"),
        _expr(cols, ["longitude"], "DOUBLE", "longitude"),
        _expr(cols, ["minimum_nights"], "INTEGER", "minimum_nights"),
        _expr(cols, ["maximum_nights"], "INTEGER", "maximum_nights"),
        _expr(cols, ["number_of_reviews"], "INTEGER", "number_of_reviews"),
        _expr(cols, ["number_of_reviews_ltm"], "INTEGER", "number_of_reviews_ltm"),
        _expr(cols, ["review_scores_rating"], "DOUBLE", "review_scores_rating"),
        _expr(cols, ["reviews_per_month"], "DOUBLE", "reviews_per_month"),
        _expr(cols, ["availability_30"], "INTEGER", "availability_30"),
        _expr(cols, ["availability_60"], "INTEGER", "availability_60"),
        _expr(cols, ["availability_90"], "INTEGER", "availability_90"),
        _expr(cols, ["availability_365"], "INTEGER", "availability_365"),
        _expr(cols, ["occupancy_rate_pct"], "DOUBLE", "occupancy_rate_pct"),
        _expr(cols, ["estimated_annual_revenue"], "DOUBLE", "estimated_annual_revenue"),
        _expr(cols, ["estimated_monthly_revenue"], "DOUBLE", "estimated_monthly_revenue"),
        _expr(cols, ["avg_booked_price"], "DOUBLE", "avg_booked_price"),
        _expr(cols, ["price_per_bedroom"], "DOUBLE", "price_per_bedroom"),
        _expr(cols, ["price_per_person"], "DOUBLE", "price_per_person"),
        _expr(cols, ["price_vs_neighbourhood"], "DOUBLE", "price_vs_neighbourhood"),
        _expr(cols, ["host_tenure_years"], "DOUBLE", "host_tenure_years"),
        _expr(cols, ["is_active"], "BOOLEAN", "is_active"),
        _expr(cols, ["is_professional_host"], "BOOLEAN", "is_professional_host"),
    ]

    con.execute(
        f"""
        CREATE OR REPLACE TABLE fact_listing_snapshot AS
        WITH fact_base AS (
            SELECT {", ".join(fields)}
            FROM stg_master
        )
        SELECT
            row_number() OVER (ORDER BY b.city, b.listing_id) AS listing_key,
            b.listing_id,
            h.host_key,
            p.property_key,
            n.neighbourhood_key,
            c.city_key,
            d.date_key AS snapshot_date_key,
            b.price_local,
            b.price_usd,
            b.latitude,
            b.longitude,
            b.minimum_nights,
            b.maximum_nights,
            b.number_of_reviews,
            b.number_of_reviews_ltm,
            b.review_scores_rating,
            b.reviews_per_month,
            b.availability_30,
            b.availability_60,
            b.availability_90,
            b.availability_365,
            b.occupancy_rate_pct,
            b.estimated_annual_revenue,
            b.estimated_monthly_revenue,
            b.avg_booked_price,
            b.price_per_bedroom,
            b.price_per_person,
            b.price_vs_neighbourhood,
            b.host_tenure_years,
            b.is_active,
            b.is_professional_host
        FROM fact_base b
        LEFT JOIN dim_host h ON b.host_id = h.host_id
        LEFT JOIN dim_property p ON b.listing_id = p.listing_id
        LEFT JOIN dim_neighbourhood n
            ON b.city = n.city AND b.neighbourhood_name = n.neighbourhood_name
        LEFT JOIN dim_city c ON b.city = c.city_name
        LEFT JOIN dim_date d ON b.snapshot_date = d.full_date
        WHERE b.listing_id IS NOT NULL
        """
    )


def _create_fact_calendar(con: duckdb.DuckDBPyConnection) -> None:
    """Create daily calendar fact, if staged calendar data exists."""
    if not _has_table(con, "stg_calendar"):
        con.execute(
            """
            CREATE OR REPLACE TABLE fact_calendar (
                listing_key INTEGER,
                date_key INTEGER,
                is_available BOOLEAN,
                price_local DOUBLE,
                price_usd DOUBLE,
                adjusted_price DOUBLE,
                minimum_nights INTEGER,
                maximum_nights INTEGER
            )
            """
        )
        return

    cols = _column_names(con, "stg_calendar")
    price = _raw_expr(cols, ["price_local", "price"], "DOUBLE")
    adjusted = _raw_expr(cols, ["adjusted_price_local", "adjusted_price"], "DOUBLE")
    rate_case = _city_usd_rate_case()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE fact_calendar AS
        SELECT
            f.listing_key,
            d.date_key,
            TRY_CAST(c.available AS BOOLEAN) AS is_available,
            {price} AS price_local,
            ROUND({price} * {rate_case}, 2) AS price_usd,
            {adjusted} AS adjusted_price,
            TRY_CAST(c.minimum_nights AS INTEGER) AS minimum_nights,
            TRY_CAST(c.maximum_nights AS INTEGER) AS maximum_nights
        FROM stg_calendar c
        LEFT JOIN fact_listing_snapshot f
            ON TRY_CAST(c.listing_id AS BIGINT) = f.listing_id
            AND c.city = (SELECT city_name FROM dim_city WHERE city_key = f.city_key)
        LEFT JOIN dim_date d ON TRY_CAST(c.date AS DATE) = d.full_date
        """
    )


def _create_fact_review(con: duckdb.DuckDBPyConnection) -> None:
    """Create review event fact, if staged review data exists."""
    if not _has_table(con, "stg_reviews"):
        con.execute(
            """
            CREATE OR REPLACE TABLE fact_review (
                review_id BIGINT,
                listing_key INTEGER,
                reviewer_key INTEGER,
                review_date_key INTEGER,
                comment_length INTEGER
            )
            """
        )
        return

    cols = _column_names(con, "stg_reviews")
    review_id = (
        "TRY_CAST(r.review_id AS BIGINT)"
        if "review_id" in cols
        else "TRY_CAST(r.id AS BIGINT)"
        if "id" in cols
        else "CAST(NULL AS BIGINT)"
    )
    reviewer_id = "TRY_CAST(r.reviewer_id AS BIGINT)" if "reviewer_id" in cols else "CAST(NULL AS BIGINT)"
    comments = "length(r.comments)" if "comments" in cols else "CAST(NULL AS INTEGER)"
    con.execute(
        f"""
        CREATE OR REPLACE TABLE fact_review AS
        SELECT
            COALESCE({review_id}, row_number() OVER (ORDER BY r.city, r.listing_id, r.date)) AS review_id,
            f.listing_key,
            dr.reviewer_key,
            d.date_key AS review_date_key,
            TRY_CAST({comments} AS INTEGER) AS comment_length
        FROM stg_reviews r
        LEFT JOIN fact_listing_snapshot f
            ON TRY_CAST(r.listing_id AS BIGINT) = f.listing_id
            AND r.city = (SELECT city_name FROM dim_city WHERE city_key = f.city_key)
        LEFT JOIN dim_reviewer dr ON {reviewer_id} = dr.reviewer_id
        LEFT JOIN dim_date d ON TRY_CAST(r.date AS DATE) = d.full_date
        """
    )


def _table_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Return row counts for all star-schema tables."""
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
    return {table: con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in tables}


def build_star_schema(
    city_names: list[str],
    db_path: str | Path | None = None,
    include_calendar: bool = True,
    include_reviews: bool = True,
) -> ModelingResult:
    """Build all star-schema tables in DuckDB from enriched/staging Parquet."""
    if not city_names:
        raise ValueError("At least one city is required to build the star schema.")

    city_names = [city.strip() for city in city_names if city.strip()]
    master_paths = _stage_required_enriched(city_names)
    calendar_paths = _existing_staging_files(city_names, "calendar") if include_calendar else []
    review_paths = _existing_staging_files(city_names, "reviews") if include_reviews else []

    warnings: list[str] = []
    if include_calendar and not calendar_paths:
        warnings.append("calendar_fact_empty")
    if include_reviews and not review_paths:
        warnings.append("review_fact_empty")

    con = get_connection(db_path)
    try:
        _create_parquet_view(con, "stg_master", master_paths)
        _create_parquet_view(con, "stg_calendar", calendar_paths, add_city_from_path=True)
        _create_parquet_view(con, "stg_reviews", review_paths, add_city_from_path=True)

        _create_dim_city(con, city_names)
        _create_dim_date(con)
        _create_dim_host(con)
        _create_dim_property(con)
        _create_dim_neighbourhood(con)
        _create_dim_reviewer(con)
        _create_fact_listing_snapshot(con)
        _create_fact_calendar(con)
        _create_fact_review(con)

        counts = _table_counts(con)
        logger.info("Star schema built at %s: %s", db_path or get_db_path(), counts)
        return ModelingResult(
            db_path=str(Path(db_path) if db_path is not None else get_db_path()),
            cities=city_names,
            table_counts=counts,
            warnings=warnings,
        )
    finally:
        con.close()


def _parse_named_queries(sql_text: str) -> dict[str, str]:
    """Parse ``-- name: query_name`` blocks from analytical_queries.sql."""
    matches = list(re.finditer(r"^--\s*name:\s*([a-zA-Z0-9_]+)\s*$", sql_text, flags=re.MULTILINE))
    queries: dict[str, str] = {}
    for idx, match in enumerate(matches):
        name = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(sql_text)
        body = sql_text[start:end].strip()
        statements = [part.strip() for part in body.split(";") if part.strip()]
        if statements:
            queries[name] = statements[0]
    return queries


def _fetch_dicts(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    """Execute SQL and return rows as dictionaries without optional Arrow deps."""
    cursor = con.execute(sql)
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def run_analytical_queries(
    query_name: str | None = None,
    sql: str | None = None,
    db_path: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run one ad-hoc SQL query or named analytical query against DuckDB."""
    if not query_name and not sql:
        raise ValueError("Provide either query_name or sql.")

    con = get_connection(db_path)
    try:
        if sql:
            return {"custom": _fetch_dicts(con, sql)}

        query_path = Path(__file__).resolve().parent.parent / "sql" / "analytical_queries.sql"
        queries = _parse_named_queries(query_path.read_text(encoding="utf-8"))
        if query_name not in queries:
            available = ", ".join(sorted(queries))
            raise KeyError(f"Unknown query '{query_name}'. Available: {available}")
        return {query_name: _fetch_dicts(con, queries[query_name])}
    finally:
        con.close()
